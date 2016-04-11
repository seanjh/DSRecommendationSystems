import sys


def parse_args(help_prompt, args):
    if len(args) > 1:
        if args[1] == '-h' or args[1] == '--help':
            print(help_prompt)
            sys.exit(0)
        elif len(args) == 3 and args[1] == '-u':
            return {"user_id": args[2]}
        elif len(args) == 2:
            return {"filename": args[1]}

    print(help_prompt)
    sys.exit("Error: Invalid arguments\n")
