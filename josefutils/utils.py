import torch, sys, tqdm, datetime, time, re, os, inspect, argparse


def get_device(force_cpu: bool = False):
    if force_cpu:
        return torch.device("cpu")
    elif sys.platform.lower() == 'darwin':  # macOS
        return torch.device("mps")
    else:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def tqdm_estimated_finish_time(t: tqdm.std.tqdm) -> str:
    rate = t.format_dict["rate"]
    remaining = max(t.total - t.n - 1, 0) / rate if rate and t.total else 0  # Seconds*
    eft = datetime.datetime.fromtimestamp(time.time() + remaining)
    today = datetime.datetime.now().day
    eft_str = eft.strftime("EFT %H:%M" if eft.day == today else "EFT %a %H:%M\033[0m")
    return eft_str


def print_exception(e: Exception):
    caller = inspect.currentframe().f_back.f_code.co_name
    print(f"❌\033[1;31mException from \033[35m{caller}\033[1;31m: {str(e)}\033[0m❌")


def safe_remove(file, silent: bool = True) -> bool:
    try:
        os.remove(file) if os.path.isfile(file) else ()
        return True
    except Exception as e:
        print_exception(e) if not silent else ()
        return False


def ansi_only_str(line: str) -> str:
    ansi_escapes = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return re.sub(ansi_escapes, '', line)


def get_file_contents_v4(file_name: str, remove_comments: bool = False, ansi_only: bool = False) -> [str]:
    """
    :param file_name: The file to read as text file
    :param remove_comments: comments with '#' will be removed if set
    :param ansi_only: remove non ansi characters which are most likely colour codes
    :return: a list of str
    """
    try:
        ansi_escapes = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        with open(file_name, 'r') as f:
            lines = [line.strip() for line in f]  # remove left and right white spaces and '\n'
            lines = [re.sub(ansi_escapes, '', line).strip() for line in lines] if ansi_only else lines
            lines = [re.sub("#.*", '', line).strip() for line in lines] if remove_comments else lines
            lines = [line for line in lines if line]  # exclude empty lines
            return lines
    except Exception as e:
        #print(f"\033[1;31mException: {str(e)}\033[0m")
        return []


def save_to_file(file_name: str, contents: list[str] = [], append=False):
    f = open(file_name, 'a') if append else open(file_name, 'w')
    [f.write(line + '\n') for line in contents]
    f.close()


def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    if x < 0.50 or x > 0.99:
        raise argparse.ArgumentTypeError("%r not in range [0.50, 0.99]" % (x,))
    return x

