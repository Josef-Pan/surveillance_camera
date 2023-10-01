import imagehash, datetime, tqdm, time, os, shutil
import torch, cv2
from PIL import Image
from collections import defaultdict
import numpy as np
from .utils import tqdm_estimated_finish_time, safe_remove, print_exception
from .image_functions import glob_image_files, mark_exif_with_lens_make, get_exif_with_lens_make


class SimilarImages:
    def __init__(self, images_dirs: str, mse_threshold: float,
                 calculate_mse: bool = True,
                 exifs_to_exclude: list[str] = None,
                 exclude_file_name_pattern=None) -> None:
        self.image_folders = images_dirs
        self.mse_threshold = mse_threshold
        self.exifs_to_exclude = exifs_to_exclude
        self.exclude_file_name_pattern = exclude_file_name_pattern
        self.average_width = None
        self.average_height = None
        self.total = None
        self.similar_images_count = None
        self.groups = self.probable_similar_picture_groups()
        if calculate_mse:
            self.similar_images_lists = self.calc_similar_images_lists()

    @staticmethod
    def cosine_similarity(img_file1: str, img_file2: str) -> float:
        image1 = Image.open(img_file1).convert("L")
        image2 = Image.open(img_file2).convert("L")
        width, height = max(image1.size[0], image2.size[0]), max(image1.size[1], image2.size[1])
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))
        image1_array = np.array(image1)
        image2_array = np.array(image2)
        image1_array = torch.tensor(image1_array, dtype=float)
        image2_array = torch.tensor(image2_array, dtype=float)
        similarity = torch.nn.functional.cosine_similarity(image1_array, image2_array).std().item()
        return similarity

    def calculate_mse(self, img_file1: str, img_file2: str, return_str: bool = False) -> float:
        image1 = Image.open(img_file1)
        image2 = Image.open(img_file2)
        if image1.size != image2.size:
            return self.mse_threshold * 2
        image1 = image1.convert("L")
        image2 = image2.convert("L")
        pixels1 = np.array(image1)
        pixels2 = np.array(image2)
        mse = np.mean((pixels1 - pixels2) ** 2)
        mse = mse * self.average_width * self.average_height / (pixels1.shape[0] * pixels1.shape[1])
        if return_str:
            return f"\033[32m{mse:8.2f}\033[0m" if mse > self.mse_threshold else f"\033[31m{mse:8.2f}\033[0m"
        else:
            return mse

    @staticmethod
    def hamming_distance(hash_1: str, hash_2: str) -> int:
        """
        Calculates the Hamming distance between two hashes.
        """
        assert len(hash_1) == len(hash_2)
        distance = sum(c1 != c2 for c1, c2 in zip(hash_1, hash_2))
        return distance

    def __file_should_be_excluded_by_pattern(self, file_name: str) -> bool:
        if self.exclude_file_name_pattern is not None:
            file_name_base, extension = os.path.splitext(file_name)
            return True if file_name_base.endswith(self.exclude_file_name_pattern) else False
        else:
            return False

    def __file_should_be_excluded_by_exif(self, file_name: str) -> bool:
        lens_make = get_exif_with_lens_make(file_name)
        if lens_make and lens_make in self.exifs_to_exclude:
            return True
        else:
            return False

    def probable_similar_picture_groups(self) -> list:
        """
        Generates a list of possible groups of similar pictures.
        Returns:
            list: A list containing groups of similar pictures.
        """
        hash_index = defaultdict(list)
        all_images = []
        for folder in self.image_folders:
            images_of_folder = glob_image_files(folder)
            all_images.extend(images_of_folder)
        all_images = [image for image in all_images if not self.__file_should_be_excluded_by_pattern(image)]
        all_images = [image for image in all_images if not self.__file_should_be_excluded_by_exif(image)]
        self.total = len(all_images)
        str_date = datetime.datetime.now().strftime("%m-%d %H:%M")
        trange_obj = tqdm.tqdm(all_images, bar_format='{l_bar}{bar:40}{r_bar}')
        total_width, count, total_height = 0, 0, 0
        for image_path in trange_obj:
            try:
                img_np = Image.open(image_path)
            except Exception as e:
                safe_remove(image_path)
                print_exception(e)
                continue
            total_width += img_np.size[0]
            total_height += img_np.size[1]
            count += 1
            self.average_width, self.average_height = total_width / count, total_height / count
            try:
                img_hash = str(imagehash.average_hash(img_np))
            except Exception as e:
                print_exception(e)
                continue
            hash_index[img_hash].append(image_path)
            possible_similar = sum([len(v) - 1 for v in hash_index.values() if len(v) > 1])
            eft = tqdm_estimated_finish_time(trange_obj)
            bar = (f'\033[1;36m{str_date} {self.average_width:.0f}*{self.average_height:.0f} '
                   f'probable similar = \033[31m{possible_similar:04d} \033[35m{eft}\033[0m')
            trange_obj.set_description(bar)
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.001)  # For demonstration purpose, should be 0.001
        trange_obj.close()
        return [v for v in hash_index.values() if len(v) > 1]

    def calc_similar_images_lists(self):
        similar_images_lists = []
        for group in self.groups:
            mse_list_of_group = [(f, self.calculate_mse(group[0], f, return_str=False)) for f in group[1:]]
            similar_images_list = [(file, mse) for file, mse in mse_list_of_group if mse < self.mse_threshold]
            similar_images_lists.append(similar_images_list)
        self.similar_images_count = sum([len(group) for group in similar_images_lists])
        return similar_images_lists

    def calc_similar_images_lists_with_progress(self, delete_in_place=False, mark_exif_in_place=None):
        similar_images_lists = []
        groups_joined = []
        [groups_joined.extend(group) for group in self.groups]
        base_files = [group[0] for group in self.groups]
        # [0, 3135, 3176, 3322, 3332]
        base_file_indices = [groups_joined.index(base_file) for base_file in base_files]
        str_date = datetime.datetime.now().strftime("%m-%d %H:%M")
        trange_obj = tqdm.tqdm(groups_joined, bar_format='{l_bar}{bar:40}{r_bar}')
        index = 0
        assert not (delete_in_place and mark_exif_in_place)
        if delete_in_place:
            info_str = "deleted similar"
        elif mark_exif_in_place is not None:
            info_str = f"added '{mark_exif_in_place}' to similar"
        else:
            info_str = "similar"
        for file in trange_obj:
            if index in base_file_indices:
                base_file = groups_joined[index]
            else:
                mse = self.calculate_mse(base_file, file)
                if mse < self.mse_threshold:
                    similar_images_lists.append((file, mse))
                    if delete_in_place:
                        safe_remove(file)
                    elif mark_exif_in_place:
                        mark_exif_with_lens_make(file, mark_exif_in_place)
            index += 1
            eft = tqdm_estimated_finish_time(trange_obj)
            bar = (f"\033[1;36m{str_date} "
                   f"{info_str} = \033[31m{len(similar_images_lists):04d} \033[35m{eft}\033[0m")
            trange_obj.set_description(bar)
            trange_obj.refresh()  # to show immediately the update
            time.sleep(0.001)  # For demonstration purpose, should be 0.001
        trange_obj.close()
        return similar_images_lists

    def del_from_second_in_similar_picture_groups(self, printout=False) -> None:
        """
        Deletes files from the second element onwards in each group of similar pictures.
        """

        def delete_a_file(file_name: str, mse: float) -> None:
            if safe_remove(file_name, silent=True):
                print(f"\033[1;31mDeleting file: {mse:.2f} \033[0m{file_name} \033[0m") if printout else ()
            else:
                print(f"\033[1;35mDeleting file: \033[0m{file_name} failed.\033[0m") if printout else ()

        # This list was calculated already, all mse values are less than mse_threshold
        for group in self.similar_images_lists:
            [delete_a_file(file, mse) for file, mse in group]

    def rename_from_second_in_similar_picture_groups(self, symbol_add_to_file_name: str, printout=False) -> None:
        """
        We don't delete the similar files, but put an extra symbol to file name end, so that we
        can exclude process those files next time
        """

        def rename_a_file(file_name: str, mse: float) -> None:
            file_name_base, extension = os.path.splitext(file_name)
            new_file_name = file_name_base + symbol_add_to_file_name + extension
            if shutil.move(file_name, new_file_name):
                print(f"\033[1;31mRenaming file: {mse:.2f} \033[0m{file_name} \033[0m") if printout else ()

        # This list was calculated already, all mse values are less than mse_threshold
        for group in self.similar_images_lists:
            [rename_a_file(file, mse) for file, mse in group]

    def exif_mark_from_second_in_similar_picture_groups(self, exif_str_similar: str, printout=False) -> None:
        """
        We don't delete the similar files, but put an extra symbol to file name end, so that we
        can exclude process those files next time
        """

        def add_exif_to_image(file_name: str, mse: float) -> None:
            mark_exif_with_lens_make(file_name, exif_str_similar)
            print(f"Marking {file_name} @ {mse:.2f} with  exif {exif_str_similar}") if printout else ()

        # This list was calculated already, all mse values are less than mse_threshold
        for group in self.similar_images_lists:
            [add_exif_to_image(file, mse) for file, mse in group]


class SimilarImagesV2:
    def __init__(self, images_dirs: list[str], mse_threshold: float):
        self.images_dirs = images_dirs
        self.mse_threshold = mse_threshold
        self.total = None
        self.groups = self.possible_similar_picture_groups()
        self.similar_images_count = None
        self.similar_images_lists = None

    def calculate_mse(self, img_file1: str, img_file2: str) -> float:
        image1 = Image.open(img_file1)
        image2 = Image.open(img_file2)
        if image1.size != image2.size:
            return self.mse_threshold * 2
        image1 = image1.convert("L")
        image2 = image2.convert("L")
        pixels1 = np.array(image1)
        pixels2 = np.array(image2)
        return np.mean((pixels1 - pixels2) ** 2)

    def calculate_mse_trial(self, img_np_array1: np.ndarray, img_np_array2: np.ndarray) -> float:
        if img_np_array1.shape != img_np_array2.shape:
            return self.mse_threshold * 2
        gray1 = cv2.cvtColor(img_np_array1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img_np_array2, cv2.COLOR_BGR2GRAY)
        mse = np.mean((gray1 - gray2) ** 2)
        return mse

    def possible_similar_picture_groups(self) -> list:
        all_images = []
        for folder in self.images_dirs:
            images_of_folder = glob_image_files(folder)
            all_images.extend(images_of_folder)
        hash_index = defaultdict(list)
        for image_file in all_images:
            try:
                img_hash = imagehash.average_hash(Image.open(image_file))
                hash_index[img_hash].append(image_file)
            except Exception as e:
                print_exception(e)
        return [v for v in hash_index.values() if len(v) > 1]

    def calc_similar_images_lists(self):
        similar_images_lists = []
        for group in self.groups:
            mse_list_of_group = [(f, self.calculate_mse(group[0], f)) for f in group[1:]]
            similar_images_list = [(file, mse) for file, mse in mse_list_of_group if mse < self.mse_threshold]
            similar_images_lists.append(similar_images_list)
        return similar_images_lists

    def remove_duplicate_images_coarse(self):
        """
        Deletes image_np from the second element onwards in each group of similar pictures.
        This method is not accurate enough, because even very different pictures may have the same hash
        """
        files_to_remove = []
        for group in self.groups:
            files_to_remove.extend(group[1:])
        [safe_remove(file) for file in files_to_remove]

    def remove_duplicate_images_fine(self):
        """
        Deletes image_np from the second element onwards in each group of similar pictures.
        """
        self.similar_images_lists = self.calc_similar_images_lists()  # A list of lists containing similar images
        self.similar_images_count = sum([len(group) for group in self.similar_images_lists])
        [safe_remove(file) for file, mse in self.similar_images_lists]
