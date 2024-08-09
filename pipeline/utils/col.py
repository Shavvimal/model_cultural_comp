import pprint


class Col:
    """
    Print colours and format json
    Use example: f'{bcolors.OKGREEN}coloured text'
    """
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREENBG = '\033[42m\033[30m'

    def printh(self, input_str: str):
        print(f'{self.HEADER}{input_str}{self.ENDC}')

    def printb(self, input_str: str):
        print(f'{self.BLUE}{input_str}{self.ENDC}')

    def printc(self, input_str: str):
        print(f'{self.CYAN}{input_str}{self.ENDC}')

    def printg(self, input_str: str):
        print(f'{self.GREEN}{input_str}{self.ENDC}')

    def printw(self, input_str: str):
        print(f'{self.WARNING}{input_str}{self.ENDC}')

    def printf(self, input_str: str):
        print(f'{self.FAIL}{input_str}{self.ENDC}')

    def printn(self, input_str: str):
        print(f'{self.ENDC}{input_str}{self.ENDC}')

    def printbb(self, input_str: str):
        print(f'{self.BOLD}{input_str}{self.ENDC}')

    def printu(self, input_str: str):
        print(f'{self.UNDERLINE}{input_str}{self.ENDC}')

    @staticmethod
    def pprint(input_str: str):
        pprint.pprint(input_str)


p = Col()

if __name__ == '__main__':
    p.printh("Header")
    p.printb("Blue")
    p.printc("Cyan")
    p.printg("Green")
    p.printw("Warning")
    p.printf("Fail")
    p.printn("Normal")
    p.printbb("Bold")
    p.printu("Underline")
    p.pprint({"key": "value"})