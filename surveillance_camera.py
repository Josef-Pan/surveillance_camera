import torch, os, sys, glob, tqdm, time, argparse, shutil, inspect, io
import datetime, re, ctypes, requests, cv2
from dateutil.parser import parse
from PIL import Image
from enum import Enum
import numpy as np
from urllib.parse import urljoin, urlparse
from multiprocessing import Process, Value, Queue, Lock
from bs4 import BeautifulSoup
from collections import namedtuple
from josefutils import get_device, SimilarImages, get_file_contents_v4, print_exception, glob_image_files
from josefutils import save_to_file, restricted_float, tqdm_estimated_finish_time, get_exif_with_lens_make
from josefutils import mark_exif_with_lens_make, save_img_np_grayscale, safe_remove

URLTemplate = namedtuple('URLTemplate', ['url_full', 'url_base', 'path', 'hostname', 'port', 'username', 'password'])
URLFields = namedtuple('URLFields', ['http_or_https', 'fields'])
MayNotWork = "❗️"  # some functions may not work with your camera, search this symbol first


class Configurations:
    def __init__(self):
        self.working_path = os.path.abspath(os.path.dirname(sys.argv[0]))
        self.cameras_ini = os.path.join(self.working_path, "cameras.ini")  # File for camera urls and/or other settings
        self.cameras = []  # Should be provided by cmd_args or a file called cameras.ini
        self.days_keep = 14  # Days video and pictures will be kept for
        self.classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
                        5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
                        10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird",
                        15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
                        20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
                        25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
                        30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
                        35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
                        40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
                        45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
                        50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
                        55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
                        60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
                        65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave", 69: "oven",
                        70: "toaster", 71: "sink", 72: "refrigerator", 73: "book", 74: "clock",
                        75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier", 79: "toothbrush"}
        self.classes_to_detect = [0, 15, 16]  # We keep videos and pictures with human, cat and dog as in list above
        self.device = None  # Initialised after parsing cmd_args
        self.yolo_model = None  # Initialised after load_yolo_model in function main()
        self.min_yolo_confidence = None  # Minimum confidence for a yolo to detect an object, set by cmd_args
        self.line_color_human = (0x00, 0xD6, 0xE5)  # Cyan
        self.line_color_others = (0xE5, 0x00, 0xB5)  # Magenta
        self.line_width = 1  # The line_width for drawing boxes around objects detected in pictures
        self.save_dir = None  # Set by cmd_args, where to save the pictures and videos
        self.dir_images = 'images'  # The dir to save downloaded pictures under save_dir + camera hostname above
        self.dir_images_processed = 'images_processed'  # The dir to save processed pictures
        self.dir_videos = 'videos'  # The dir to save downloaded videos under save_dir + camera hostname above
        self.dir_videos_processed = 'videos_processed'  # The dir to save processed videos
        self.temp_dir_video_frames = '.temp_video_frames'  # The dir save temporary video frames inflated from video
        self.target_date_str = None  # The date string in to search in camera webpages for specific date
        # For testing similarity, for a 1920x1080 image, 120 is a proper value,
        # for a 1280x720 image, 80 is a proper value, the larger resolution, the larger the value
        self.mse_threshold = None  # The threshold for judging whether two images are similar, default 120
        self.picture_formats = None  # Picture formats to search in camera webpages to download
        self.video_formats = None  # Video formats  to search in camera webpages to download
        self.image_dir_format = None  # Format string to search in camera webpages for specific date, e.g. '%Y%m%d'
        self.download_processes = []  # Download processes started by CameraCrawler, saved here to kill at Ctrl+C
        self.image_urls = []  # image urls already downloaded, updated by program
        self.image_urls_log = 'image_urls.log'  # A base name, NOT full path
        self.video_urls = []  # video urls already downloaded, updated by program
        self.video_urls_log = 'video_urls.log'  # A base name, NOT full path
        self.video_processed = []  # Adding metadata to video is somewhat difficult, so we use a log file instead
        self.video_processed_log = 'video_processed.log'  # A base name, NOT full path
        os.environ['OPENCV_LOG_LEVEL'] = 'OFF'  # Suppress open_cv warnings when inflating frames into pictures
        os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"  # Suppress open_cv warnings when inflating frames into pictures

    def update_from_cmd_args(self, args: argparse.Namespace):
        """
        Copy all the keys/values of args into Configurations
        """
        for key, value in vars(args).items():
            setattr(self, key, value)
        self.device = get_device(force_cpu=self.cpu)
        today = datetime.date.today()
        target_date = today if args.today else (today - datetime.timedelta(days=1))
        self.target_date_str = args.date if args.date else target_date.strftime('%Y%m%d')  # 20211113
        self.__load_cameras_ini_file()
        image_urls_log = os.path.join(self.working_path, self.image_urls_log)
        self.image_urls = get_file_contents_v4(image_urls_log)
        video_processed_log = os.path.join(self.working_path, self.video_processed_log)
        self.video_processed = get_file_contents_v4(video_processed_log)
        dayskeep_time = [today - datetime.timedelta(days=item) for item in range(30)]
        dayskeep_time_str = [item.strftime(self.image_dir_format) for item in dayskeep_time]
        self.image_urls = [item for item in self.image_urls
                           if any([d in item for d in dayskeep_time_str])]  # Keep only recent logs
        self.video_processed = [item for item in self.video_processed
                                if any([d in item for d in dayskeep_time_str])]  # Keep only recent logs
        classes_to_detect = self.classes_to_detect
        self.classes_to_detect = [k for k, v in self.classes.items() if v in classes_to_detect]

        # four_dirs = [self.dir_images, self.dir_images_processed, self.dir_videos, self.dir_videos_processed]
        # [os.makedirs(os.path.join(self.save_dir, d), exist_ok=True) for d in four_dirs]

    def load_yolo_model(self):
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True, device=self.device)

    def __load_cameras_ini_file(self):
        ini_contents = get_file_contents_v4(self.cameras_ini, remove_comments=True)
        ini_dict = {line.split('=')[0].strip(): line.split('=')[-1].strip() for line in ini_contents}
        ini_dict = {k.replace('"', ''): v.replace('"', '') for k, v in ini_dict.items()}  # Remove quotes
        ini_dict = {k.replace("'", ''): v.replace("'", '') for k, v in ini_dict.items()}  # Remove quotes
        cameras = [k for k, v in ini_dict.items() if k == v]
        settings = {k: v.split() if len(v.split()) > 1 else v for k, v in ini_dict.items() if k != v}
        if cameras:
            print(f"✅\033[36mLoaded \033[0m{len(cameras)}\033[36m cameras from cameras.ini ...\033[0m")
            [print(camera) for camera in cameras]
        if isinstance(self.cameras, list):  # Not None, a list of cameras
            self.cameras.extend(cameras)
        else:
            self.cameras = cameras
        if settings:
            print(f"✅\033[36mLoaded \033[0m{len(settings.keys())}\033[36m settings from cameras.ini ...\033[0m")
            [print(f"{k} -> {v}") for k, v in settings.items()]
            [setattr(self, key, value) for key, value in settings.items()]

    def save_log_files(self):
        save_to_file(self.image_urls_log, list(self.image_urls), append=False)
        save_to_file(self.video_processed_log, list(self.video_processed), append=False)

    def remove_old_images_and_videos(self):
        today = datetime.date.today()
        days_to_keep = int(self.remove_old_files[0])
        if (len(self.remove_old_files) > 1 and
                (self.remove_old_files[1] == 'delete' or self.remove_old_files[1] == 'del')):
            real_delete = True
        else:
            real_delete = False
        dayskeep_time = [today - datetime.timedelta(days=item) for item in range(days_to_keep)]
        dayskeep_time_str = [item.strftime(self.image_dir_format) for item in dayskeep_time]
        # First level dirs are camera names or shortened urls
        sub_dirs_1st_level = [item for item in glob.glob(os.path.join(self.save_dir, '*')) if os.path.isdir(item)]
        for sub_dir in sub_dirs_1st_level:
            # Second level dirs are images, images_processed, videos, videos_processed
            sub_dirs_2nd_level = [item for item in glob.glob(os.path.join(sub_dir, '*')) if os.path.isdir(item)]
            for sub_dir_2 in sub_dirs_2nd_level:
                # Third level dirs are days in the format of cfg.image_dir_format
                sub_dirs_3rd_level = [item for item in glob.glob(os.path.join(sub_dir_2, '*')) if os.path.isdir(item)]
                will_remove = [item for item in sub_dirs_3rd_level if os.path.basename(item) not in dayskeep_time_str]
                print(f"\033[35mOld directories: {will_remove}\033[0m") if len(will_remove) else None
                if real_delete:
                    print(f"\033[1;31mDeleting {will_remove}\033[0m") if len(will_remove) else None
                    [shutil.rmtree(d) for d in will_remove]

    def __repr__(self):
        string_list = [f"",
                       f"", ]
        return '\n'.join(string_list)


class MediaType(Enum):
    IMAGES = "images"
    VIDEOS = "videos"


class CameraCrawler:
    def __init__(self, cfg: Configurations, url: str, media_type: MediaType):
        self.cfg = cfg
        self.url = url
        self.url_base = None  # like 'http://192.168.1.224', no port number, no path
        self.media_type = media_type
        # The number of download workers running in separate processes, video files too large, we assign more workers
        self.num_download_workers = 3 if media_type == MediaType.IMAGES else 6
        self.visited_urls = set()
        self.image_urls = cfg.image_urls
        self.queue = Queue()
        self.lock = Lock()
        self.username, self.password = None, None
        self.images_downloaded = Value(ctypes.c_int64, 0)

    def start(self):
        url_template = interpret_url(self.url)
        self.username, self.password = url_template.username, url_template.password
        self.url, self.url_base = url_template.url_full, url_template.url_base
        if self.media_type == MediaType.IMAGES:
            dir_images = os.path.join(self.cfg.save_dir, url_template.hostname,
                                      self.cfg.dir_images, self.cfg.target_date_str)
            os.makedirs(dir_images, exist_ok=True)
            dir_images_or_videos = dir_images
            supported_formats = self.cfg.picture_formats
            media_tag = 'img'
            media_info = 'images'
            download_videos = False
        else:
            dir_videos = os.path.join(self.cfg.save_dir, url_template.hostname,
                                      self.cfg.dir_videos, self.cfg.target_date_str)
            os.makedirs(dir_videos, exist_ok=True)
            dir_images_or_videos = dir_videos
            supported_formats = self.cfg.video_formats
            media_tag = 'video'
            media_info = 'videos'
            download_videos = True
        args = (dir_images_or_videos, self.images_downloaded, supported_formats,
                self.queue, self.lock, self.username, self.password, download_videos)
        download_processes = [Process(target=download_image_video_worker, args=args)
                              for _ in range(self.num_download_workers)]
        self.cfg.download_processes = download_processes
        [p.start() for p in download_processes]
        self.crawl_camera(depth=0, queue_=self.queue, url=self.url, media_formats=supported_formats,
                          media_tag=media_tag)
        [self.queue.put(None) for _ in range(self.num_download_workers)]
        [p.join() for p in download_processes]
        print(f"✅\033[36mDownloaded {self.images_downloaded.value} {media_info} from {self.url}.\033[0m")
        self.cfg.save_log_files()

    @staticmethod
    def fields_in_url(url):
        output = urlparse(url)
        fields = output.path.lstrip('/').split('/')
        return URLFields(output.scheme, fields)

    @staticmethod
    def can_be_parsed_as_date(date_str: str):
        try:
            date_obj = parse(date_str)
            return date_obj
        except ValueError:
            return None

    def is_link_acceptable(self, link_href: str) -> bool:
        if link_href.startswith('http://') or link_href.startswith('https://'):
            url_template = interpret_url(link_href)
            if url_template.url_base != self.url_base:  # Linked to other websites, maybe manufacturer's website
                return False
            else:
                link_href = url_template.path
        link_href = link_href.rstrip('/')
        if link_href.startswith('/'):
            date_format = self.cfg.image_dir_format
            date_obj = datetime.datetime.strptime(self.cfg.target_date_str, '%Y%m%d')
            target_date_string = date_obj.strftime(date_format)
            last_field = link_href.split('/')[-1]
            m_dot_something = re.search(r"(\.[^.]+)$", last_field)
            if m_dot_something:
                if self.media_type == MediaType.IMAGES:
                    more_extensions = self.cfg.picture_formats
                else:
                    more_extensions = self.cfg.video_formats
                if m_dot_something[1] in [".html", ".htm", ".shtml"] or m_dot_something[1] in more_extensions:
                    return True
                else:
                    return False
            elif target_date_string in link_href:
                return True
            else:
                http_or_https, fields = self.fields_in_url(link_href)
                for field in fields:
                    date_obj = self.can_be_parsed_as_date(field)
                    if date_obj is not None:
                        date_string = date_obj.strftime(date_format)
                        if date_string != target_date_string:
                            return False
                return True
        else:
            return False
        return False

    def crawl_camera(self, depth: int, queue_: Queue, url, media_formats, media_tag):
        images_found = 0
        stack = [(url, depth)]
        counter = 0
        if media_tag == 'img':
            media_info = 'I'
        else:
            media_info = 'V'
        while stack:
            counter += 1
            url, depth = stack.pop(0)
            self.visited_urls.add(url)  # Update visited_urls here
            # images_downloaded = self.images_downloaded.value
            try:
                try:
                    response = requests.get(url, auth=(self.username, self.password), timeout=30) \
                        if self.username and self.password else requests.get(url, timeout=30)
                    if response.status_code == 200:
                        response.encoding = 'utf-8'
                        soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
                    else:
                        print(f"Failed to access the page. Status code: {response.status_code}")
                        continue
                except Exception as e:
                    print_exception(f"{str(e)}@\033[36m{url}\033[0m")
                    continue
                media_tags = soup.find_all(media_tag)
                for tag in media_tags:
                    media_src = tag.get('src')
                    if media_src:
                        media_url = urljoin(url, media_src)
                        if media_url not in self.image_urls and self.cfg.target_date_str in media_url:
                            # MayNotWork ❗️, sometimes the link may have no date string
                            self.image_urls.append(media_url)
                            images_found += 1
                            queue_.put(media_url)

                if sys.platform.lower() == 'darwin':
                    qsize = -1  # macOS doesn't have a queue size
                else:
                    qsize = queue_.qsize()
                time.sleep(qsize * 0.1) if qsize > 10 else ()
                link_tags = soup.find_all('a', href=True)
                link_tags = [link for link in link_tags if
                             link['href'] and link['href'] and self.is_link_acceptable(link['href'])]
                link_tags = [e for idx, e in enumerate(link_tags) if
                             e['href'] not in [ee['href'] for ee in link_tags][:idx]]
                current_depth_urls = [u for u, d in stack if d == depth]
                len_current_depth = len(current_depth_urls)
                link_tags = [e for e in link_tags if urljoin(url, e['href']) not in self.visited_urls]
                fmt_string = (f"\033[36m{len_current_depth}/\033[1;36m{len(stack)}, Q: {qsize:2d}\033[0m "
                              f"{media_info}: \033[35m{images_found} \033[0m"  # Images found
                              f"D: \033[36m{self.images_downloaded.value}\033[0m")  # Images downloaded
                now = f"\033[35m{datetime.datetime.now().strftime(f'%Y%m%d-%H:%M:%S')}\033[0m"
                digits_of_depth = max(len(str(depth)), len(str(depth + 1)))
                info_str = (f"{now}(\033[36m{counter:<2d}\033[0m)Crawling:\033[32m{url[:60]:60}\033[0m\033[1;32m"
                            f"({depth:<{digits_of_depth}d}) \033[0m{fmt_string}")
                print(info_str)
                for link_tag in link_tags:
                    link_href = link_tag['href']
                    new_url = urljoin(url, link_href)
                    if any([link_href.endswith(f) for f in media_formats]):  # Direct link of pictures
                        if new_url not in self.image_urls and self.cfg.target_date_str in new_url:
                            # MayNotWork ❗️, sometimes the link may have no date string
                            self.image_urls.append(new_url)
                            images_found += 1
                            queue_.put(urljoin(url, link_href))
                    else:
                        if self.cfg.verbose:
                            now = f"\033[35m{datetime.datetime.now().strftime(f'%Y%m%d-%H:%M:%S')}\033[0m"
                            link_output = link_href.replace(self.url.rstrip('/'), '')
                            print(f"{now}(\033[36m{counter:<2d}\033[0m)Adding  :{link_output[:60]:60}\033[1;36m"
                                  f"({depth + 1:<{digits_of_depth}d}) \033[0m{fmt_string}")
                        stack.append((new_url, depth + 1))
            except Exception as e:
                print_exception(e)
        if sys.platform.lower() == 'darwin':
            return
        while queue_.qsize():
            time.sleep(5)
            fmt_string = (f"\033[36m{len_current_depth}/\033[1;36m{len(stack)}, Q: {queue_.qsize():2d}\033[0m "
                          f"{media_info}: \033[35m{images_found} \033[0m"  # Images found
                          f"D: \033[36m{self.images_downloaded.value}\033[0m")  # Images downloaded
            now = f"\033[35m{datetime.datetime.now().strftime(f'%Y%m%d-%H:%M:%S')}\033[0m"
            digits_of_depth = max(len(str(depth)), len(str(depth + 1)))
            info_str = (f"{now}(\033[36m{counter}\033[0m)Crawling:\033[32m{url[:60]:60}\033[0m\033[1;32m"
                        f"({depth:<{digits_of_depth}d}) \033[0m{fmt_string}")
            print(info_str)


class CameraPicturesVideos:
    def __init__(self, cfg: Configurations, url: str):
        self.cfg = cfg
        self.url = url

    def download_pictures(self):
        print(f"🔷\033[36mDownloading \033[0mimages\033[36m from {self.url}")
        camera_crawler = CameraCrawler(self.cfg, self.url, media_type=MediaType.IMAGES)
        camera_crawler.start()

    def download_videos(self):
        print(f"🔷\033[36mDownloading \033[0mvideos\033[36m from {self.url}")
        camera_crawler = CameraCrawler(self.cfg, self.url, media_type=MediaType.VIDEOS)
        camera_crawler.start()

    def remove_similar_images(self):
        url_template = interpret_url(self.url)
        dir_images = os.path.join(self.cfg.save_dir, url_template.hostname, self.cfg.dir_images,
                                  self.cfg.target_date_str)
        print(f"🔷\033[36mMarking similar images in {dir_images} @MSE:\033[36m{self.cfg.mse_threshold:.0f}\033[0m")
        exif_str_similar = self.cfg.exif_str_similar  # We ignore files already marked as very similar to others
        exif_str_processed = self.cfg.exif_str_processed
        similar_images = SimilarImages(images_dirs=[dir_images], mse_threshold=self.cfg.mse_threshold,
                                       calculate_mse=False, exifs_to_exclude=[exif_str_similar, exif_str_processed])
        similar_images.calc_similar_images_lists_with_progress(mark_exif_in_place=exif_str_similar)

    def __file_should_be_excluded_by_exif(self, file_name: str) -> bool:
        lens_make = get_exif_with_lens_make(file_name)
        if lens_make and lens_make in [self.cfg.exif_str_similar, self.cfg.exif_str_processed]:
            return True
        else:
            return False

    def draw_boxes_and_save_to_processed(self, image_file_src, image_file_dst, boxes):
        image = Image.open(image_file_src).convert('RGB')
        image = np.asarray(image).astype(np.uint8)
        line_width = self.cfg.line_width
        for idx in range(boxes.shape[0]):
            line_color = self.cfg.line_color_human if boxes[idx][-1] == 0 else self.cfg.line_color_others
            x1, y1, x2, y2 = int(boxes[idx][0]), int(boxes[idx][1]), int(boxes[idx][2]), int(boxes[idx][3])
            image[y1:y2, max(x1 - line_width, 0): min(x1 + line_width, image.shape[1] - 1)] = line_color
            image[y1:y2, max(x2 - line_width, 0): min(x2 + line_width, image.shape[1] - 1)] = line_color
            image[max(y1 - line_width, 0): min(y1 + line_width, image.shape[0] - 1), x1:x2] = line_color
            image[max(y2 - line_width, 0): min(y2 + line_width, image.shape[0] - 1), x1:x2] = line_color
        mark_exif_with_lens_make(image_file_src, self.cfg.exif_str_processed)
        image_dst = Image.fromarray(image)
        image_dst.save(image_file_dst)

    def process_pictures(self):
        """
        For pictures already removed duplicates with special exif information and exclude the processed with
        another special exif information. We now only need to process very few pictures.
        For each picture we detect objects of interest in cfg.classes_to_detect, and then draw boxes around the
        object with special colours. Default colours are cyan for human, and magenta for others.
        :return:
        """
        url_template = interpret_url(self.url)
        dir_images_processed = os.path.join(self.cfg.save_dir, url_template.hostname, self.cfg.dir_images_processed,
                                            self.cfg.target_date_str)
        os.makedirs(dir_images_processed, exist_ok=True)
        dir_images = os.path.join(self.cfg.save_dir, url_template.url_base, self.cfg.dir_images,
                                  self.cfg.target_date_str)
        all_images = glob_image_files(dir_images)
        # We exclude images with exif tags __similar__ and __processed__ which means they are already processed or
        # already marked as similar images
        all_images = [image for image in all_images if not self.__file_should_be_excluded_by_exif(image)]
        class_names = [self.cfg.classes[k] for k in self.cfg.classes_to_detect]
        if len(all_images):
            print(f"🔷\033[36mProcessing \033[35m{len(all_images)}\033[0m \033[36mimages in {dir_images} "
                  f"with {class_names} ...\n"
                  f"Results can be found in \033[0m{dir_images_processed}")
        else:
            print(f"✅\033[36mAll images in {dir_images} are already processed.")
            return
        files_kept = 0
        str_now = datetime.datetime.now().strftime("\033[36m%m-%d %H:%M\033[0m")
        trange_obj = tqdm.tqdm(all_images, bar_format='{l_bar}{bar:30}{r_bar}', position=0, leave=True)
        for file in trange_obj:
            boxes = self.yolo_detect(file)
            if boxes is not None and boxes.shape[0] > 0:
                image_file_dst = os.path.join(dir_images_processed, os.path.basename(file))
                self.draw_boxes_and_save_to_processed(file, image_file_dst, boxes)
                files_kept += 1
            else:
                # Even though there are no boxes, we still mark the file as processed
                # so that we won't process it again
                mark_exif_with_lens_make(file, self.cfg.exif_str_processed)
            eft = f"\033[35m{tqdm_estimated_finish_time(trange_obj)}\033[0m"
            bar = f"{str_now} \033[36mKept: {files_kept:5d} \033[0m {eft}"
            trange_obj.set_description(bar)
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.01)
        trange_obj.close()

    def yolo_detect(self, image_file_or_np):  # return boxes if picture has cfg.classes_to_detect
        minor_errors = ['image file is truncated', 'cannot identify image file',
                        'tuple index out of range',
                        'No such file or directory']
        try:
            result = self.cfg.yolo_model(image_file_or_np)
        except Exception as e:
            error_str = str(e.args)
            if any([item in error_str for item in minor_errors]):
                return False
            else:  # Critical Error
                print_exception(e)
                sys.exit(202)
        boxes = result.xyxy[0]
        filter_array = np.isin(boxes[:, -1], self.cfg.classes_to_detect)
        boxes = boxes[filter_array]  # in cfg.classes_to_detect
        boxes = boxes[boxes[:, -2] > self.cfg.min_yolo_confidence]  # confidence good enough
        if boxes.shape[0]:  # has no person or other classes to keep
            return boxes
        else:
            return None

    @staticmethod
    def save_video_frames(video_file, tmp_dir):
        [safe_remove(file) for file in glob.glob(os.path.join(tmp_dir, '*'))]
        cap = cv2.VideoCapture(video_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(frame_count):
            ret, frame = cap.read()
            if ret and frame is not None:
                file_name = os.path.join(tmp_dir, f"{i:08d}.jpg")
                save_img_np_grayscale(file_name, frame)
        cap.release()
        return frame_count

    def process_videos(self):
        """
        For video files, we can't remove similar frames the same way as pictures. We use cv2 to inflate the video
        frames into a temporary picture set. Then we can remove duplicates, and detect the temporary pictures.
        If any object of interest is found. We just return and save this video file to processed directory.
        :return:
        """
        url_template = interpret_url(self.url)
        dir_videos_processed = os.path.join(self.cfg.save_dir, url_template.hostname, self.cfg.dir_videos_processed,
                                            self.cfg.target_date_str)
        os.makedirs(dir_videos_processed, exist_ok=True)
        temp_video_dir = os.path.join(self.cfg.save_dir, self.cfg.temp_dir_video_frames)
        os.makedirs(temp_video_dir, exist_ok=True)
        dir_videos = os.path.join(self.cfg.save_dir, url_template.hostname, self.cfg.dir_videos,
                                  self.cfg.target_date_str)
        all_videos = glob_image_files(dir_videos, formats=self.cfg.video_formats)
        all_videos = [f for f in all_videos if f not in self.cfg.video_processed]
        if len(all_videos):
            class_names = [self.cfg.classes[k] for k in self.cfg.classes_to_detect]
            print(f"🔷\033[36mProcessing \033[35m{len(all_videos)}\033[0m \033[36mvideos in {dir_videos} "
                  f"with {class_names} ...\n"
                  f"Results can be found in \033[0m{dir_videos_processed}")
        else:
            print(f"✅\033[36mAll videos in {dir_videos} are already processed.")
            return
        for idx, video_file in enumerate(all_videos):
            print(f"🔷🔷\033[36mProcessing {video_file} --> {idx + 1}/{len(all_videos)}\033[0m")
            self.save_video_frames(video_file, temp_video_dir)
            similar_images = SimilarImages(images_dirs=[temp_video_dir], calculate_mse=False,
                                           mse_threshold=self.cfg.mse_threshold)
            similar_images.calc_similar_images_lists_with_progress(delete_in_place=True)
            self.cfg.video_processed.append(video_file)
            save_to_file(self.cfg.video_processed_log, self.cfg.video_processed, append=False)  # A little bit ugly
            unique_images = glob_image_files(temp_video_dir)
            for image in unique_images:
                boxes = self.yolo_detect(image)
                if boxes is not None:
                    print(f"✅\033[31mFound \033[0m{len(boxes)}\033[31m objects of interest in \033[0m{video_file}")
                    shutil.copy(video_file, os.path.join(dir_videos_processed, os.path.basename(video_file)))
                    break
        self.cfg.save_log_files()

    def __repr__(self):
        string_list = [f"",
                       f"", ]
        return '\n'.join(string_list)


def is_media_already_downloaded(dir_media, media_basename):
    all_medias = glob_image_files(dir_media)
    all_medias = [os.path.basename(f) for f in all_medias]
    if media_basename in all_medias:
        return True
    else:
        return False


def download_image(img_url, username, password, dir_images, supported_formats):
    """
    Download an image from a given URL and save it to the specified directories.
    Args:
        :param dir_images: The directory to save the image.
        :param username: The username needed to access the camera.
        :param password: The password needed to access the camera.
        :param img_url: The URL of the image to download.
        :param supported_formats: List of supported image formats defined in Configurations.
    Returns:
        bool: True if the image is downloaded and saved successfully, False otherwise.
    """
    # Check if the image URL is provided
    if not img_url:
        return False
    # Extract the image name from the URL
    img_name = img_url.split('?')[0].split('/')[-1]
    # Extract the base name and extension of the image
    base, extension = os.path.splitext(os.path.basename(img_name))
    # Check if the image extension is supported
    if extension.lower() not in supported_formats:
        return False
    if is_media_already_downloaded(dir_images, img_name):
        # print(f"\033[31m{img_name}\033[0m is already downloaded.")
        return False
    # Create the full path for the image and the large version of the image
    img_full_path = os.path.join(dir_images, img_name)
    try:
        # Send a request to download the image
        response = requests.get(img_url, auth=(username, password), timeout=15) \
            if username and password else requests.get(img_url, timeout=15)
        # Check if the request was successful
        if response.status_code == 200:
            # Get the content type of the response
            content_type = response.headers.get('Content-Type', '')
            # Check if the response contains an image
            if content_type.startswith('image/'):
                # Get the image data
                image_data = response.content
                try:
                    # Try to open and convert the image data
                    img_np = Image.open(io.BytesIO(image_data)).convert('RGB')
                    img_np.save(img_full_path)
                    return True
                except Exception as e:
                    print_exception(e)
                    return False
        return False
    except Exception as e:
        print_exception(e)
        return False


def download_video(video_url, username, password, dir_videos, supported_formats):
    """
    Download an image from a given URL and save it to the specified directories.
    Args:
        :param dir_videos: The directory to save the videos.
        :param username: The username needed to access the camera.
        :param password: The password needed to access the camera.
        :param video_url: The URL of the video to download.
        :param supported_formats: List of supported video formats defined in Configurations.
    Returns:
        bool: True if the image is downloaded and saved successfully, False otherwise.
    """
    # Check if the image URL is provided
    if not video_url:
        return False
    # Extract the image name from the URL
    video_name = video_url.split('?')[0].split('/')[-1]
    # Extract the base name and extension of the image
    base, extension = os.path.splitext(os.path.basename(video_name))
    # Check if the image extension is supported
    if extension.lower() not in supported_formats:
        print(f"\033[31m{video_name} not in supported formats {supported_formats}")
        return False
    if is_media_already_downloaded(dir_videos, video_name):
        return False
    # Create the full path for the image and the large version of the image
    video_full_path = os.path.join(dir_videos, video_name)
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        with open(video_full_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
            return True
    except Exception as e:
        print_exception(e)
        return False


def download_image_video_worker(*args):
    """
    This function runs as a separate PROCESS
    Args:
        *args: Variable-length argument list. Expected arguments in order:
            - dir_images (str): The directory to save the downloaded images.
            - images_downloaded (Value): A shared value to track the number of downloaded images.
            - connection (Connection): A connection object to receive image URLs.
    Returns:
        None
    """
    dir_images_or_videos: str = args[0]
    media_downloaded: Value = args[1]
    supported_formats = args[2]
    connection: Queue = args[3]
    lock: Lock = args[4]
    username: str = args[5]
    password: str = args[6]
    will_download_videos = args[7]
    while True:
        try:
            image_video_url = connection.get(block=True)
        except Exception as e:
            print_exception(e)
            continue
        try:
            if will_download_videos:
                if download_video(image_video_url, username, password, dir_images_or_videos, supported_formats):
                    with lock:
                        media_downloaded.value += 1
            else:
                if download_image(image_video_url, username, password, dir_images_or_videos, supported_formats):
                    with lock:
                        media_downloaded.value += 1
        except Exception as e:
            print_exception(e)
            continue
        if image_video_url is None:  # Signal to break the loop when all images have been processed
            break


def interpret_url(url: str) -> URLTemplate:
    """
    "http://admin:mypassword@192.168.1.224:8080/sd/20230928/images019/search?abcde"
    url_full = 'http://192.168.1.224:8080/sd/20230928/images019/search?abcde'
    url_base = 'http://192.168.1.224'
    path = '/sd/20230928/images019/search'
    hostname = 192.168.1.224
    port = 8080
    user_name = 'admin'
    pass_word = 'mypassword'
    :param url:
    :return:
    """
    username, password = None, None
    m = re.search("(http|https)://(.*:.*)@", url)
    if m:
        username, password = m[2].split(':')[0], m[2].split(':')[1]
        url_full = re.sub("//.*:.*@", "//", url)
    else:
        url_full = url
    output = urlparse(url_full)
    url_base = output.scheme + "://" + output.hostname
    return URLTemplate(url_full, url_base, output.path, output.hostname, output.port, username, password)


def early_exit_by_ctrl_c(cfg: Configurations):
    print("Ctrl +C was pressed ...")
    this_func = inspect.currentframe().f_code.co_name
    pids = [p.pid for p in cfg.download_processes if p is not None]
    [p.terminate for p in cfg.download_processes if p is not None]
    cfg.save_log_files()
    print(f"\033[35m{this_func}\033[0m terminated download process with \033[36m{pids}\033[0m as well.")
    sys.exit(201)


def parse_arguments():
    helps = {
        "mse_threshold": ("Threshold to test picture similarity, for a picture of 1920x1080, mse 120 means "
                          "the two pictures are very very close to each other, with "
                          "sum((dot_in_picture1-dot_in_picture2)**2) ==120 among 1920*1080."),
        'save_dir': ("The directory to save the raw/processed pictures and videos. This is a \033[1;31mMUST\033[0m. "
                     "For convenience, you may add a line to your cameras.ini "
                     "\033[35msave_dir = /the_real_path_of_save_dir\033[0m. Actually you may put most other "
                     "settings with string type(not int, not float etc.) in this file in the same way as well."),
        'cameras': ("A list of cameras, \033[1;31mMUST be with username and password provided.\033[0m. Since "
                    "this can be very long, a file called 'cameras.ini' "
                    "can be provided in the working directory of this program. Each line is a camera. "
                    "For example, http://admin:password@192.168.1.201:80/sd/, "
                    "http://admin:password@192.168.1.202:81/sd/, where /sd/ refers to the SD card. \033[1;31mPlease "
                    "noted that your model url may be different from this format\033[0m. Try with "
                    "\033[35mcurl http://admin:password@192.168.1.201:80/sd/ | html2text\033[0m to see if it works."),
        'image_dir_format': ("Image dir format of cameras, e.g. %%Y%%m%%d. Different camera models may have different "
                             "dir formats. To make sure you set the correct format, use "
                             "\033[35mcurl http://admin:password@192.168.1.201:80/sd/ | html2text\033[0m and make "
                             "sure it displays a list of directories. Or you may use web browser to and "
                             "find the 'view sd card' option of the camera."),
        'classes_to_detect': ("A list of classes to detect, with default as ['person', 'cat', 'dog']. "
                              "Yo may add custom classes, eg. 'car', but make sure they are supported. Use "
                              "\033[35m--list_classes\033[0m to see all classes supported"),
        'date': "Process data of a specific date in ISO format %%Y%%m%%d, e.g. 20230217, 20231231 etc.",
        'today': ("Process today's data. By default It will process \033[35myesterday's data\033[0m, because most "
                  "likely today's data is not complete yet. Please be noted that if your camera's SD card does "
                  "not have enough space, yesterday's data may be overwritten"),
        'exif_str_similar': "A string to be added to the exif of the image for similar images",
        'exif_str_processed': "A string to be added to the exif of the image for exif_str_processed images",
        'remove_old_files': ("Remove files older than \033[35mN\033[0m days. If only one parameter is specified, "
                             "it will only show the old picture/video directories. If a second parameter is provided, "
                             "and second parameter is 'del or 'delete', they will be removed"),
    }
    parser = argparse.ArgumentParser(description='Keep camera video/photo with human beings dogs and cats',
                                     prog='shrink-camera', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-V', '--version', action='version', version='%(prog)s 2.70')
    parser.add_argument("-C", "--cameras", help=helps['cameras'], nargs='+', type=str)
    parser.add_argument('-S', "--save_dir", help=helps['save_dir'], type=str)
    parser.add_argument('-CD', "--classes_to_detect", nargs='+', help=helps['classes_to_detect'], type=str,
                        default=['person', 'cat', 'dog'])
    parser.add_argument('-Y', "--min_yolo_confidence", metavar='0.50-0.99', help="Minimum confidence to detect objects",
                        default=0.80, type=restricted_float)
    parser.add_argument("--picture_formats", nargs='+', help="Picture formats of cameras", type=str,
                        default=['.jpg', '.png', '.jpeg'])
    parser.add_argument("--video_formats", nargs='+', help="Video formats of cameras", type=str,
                        default=['.avi', '.mp4'])
    parser.add_argument("--image_dir_format", help=helps['image_dir_format'], type=str, default='%Y%m%d')
    parser.add_argument("--exif_str_similar", help=helps['exif_str_similar'], type=str, default='__similar__')
    parser.add_argument("--exif_str_processed", help=helps['exif_str_processed'], type=str, default='__processed__')
    parser.add_argument('-MT', "--mse_threshold", metavar='FLOAT', help=helps['mse_threshold'], default=120, type=float)
    parser.add_argument("-VB", "--verbose", help="Show verbose information", action="store_true")
    parser.add_argument("--cpu", help="Force using of CPU even CUDA is available", action="store_true")
    parser.add_argument("-PO", "--process_only", help="Do not download, only process images/videos",
                        action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date", help=helps['date'], type=str)
    group.add_argument("--today", help=helps['today'], action="store_true")
    group.add_argument("-LC", "--list_classes", help="List all available classes we can detect", action="store_true")
    group.add_argument("-RO", "--remove_old_files", nargs='+', help=helps['remove_old_files'])
    return parser.parse_args()


def main(cfg: Configurations):
    cmd_args = parse_arguments()
    cfg.update_from_cmd_args(cmd_args)
    if cmd_args.list_classes:
        print(cfg.classes)
        return
    elif cmd_args.remove_old_files is not None:
        cfg.remove_old_images_and_videos()
        return
    cfg.load_yolo_model()
    camera_pictures_list = [CameraPicturesVideos(cfg, url=camera) for camera in cfg.cameras]
    if not cmd_args.process_only:
        # First step, download all pictures
        for cam_pic in camera_pictures_list:
            cam_pic.download_pictures()

        # 2nd step, download all videos
        for cam_vid in camera_pictures_list:
            cam_vid.download_videos()

    # 3rd step, remove similar pictures
    for cam_pic in camera_pictures_list:
        cam_pic.remove_similar_images()

    # 4th step, detect human, dog, cat etc
    for cam_pic in camera_pictures_list:
        cam_pic.process_pictures()

    # 5th step, remove similar frames in videos and detect human, dog, cat
    for cam_vid in camera_pictures_list:
        cam_vid.process_videos()


if __name__ == "__main__":
    configurations = Configurations()
    try:
        main(configurations)
    except KeyboardInterrupt:
        early_exit_by_ctrl_c(configurations)
