DSEGait_LOGO = """

"""

def nice_print(msg, last=False):
    print()
    print("\033[0;34m" + msg + "\033[0m")
    if last:
        print()

def cli_logo():
    nice_print(DSEGait_LOGO)
