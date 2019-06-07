import re
import argparse

def daide_script_to_csv(filename):
    search_replaces = [(r"^.*?\[(\d+)\] ?(request:\[.*?\]).*|"          # \1 \2
                        r"^.*?\[(\d+)\] ?(response:\[.*?\]).*|"         # \3 \4
                        r"^.*?\[(\d+)\] ?(notification:\[.*?\]).*|"     # \5 \6
                        r"^.*?(=== NEW PHASE ===).*|"                   # \7
                        r"^.*?([A-Z]\d{4}[A-Z]).*|"                     # \8
                        r".*",
                        r"\1\2\3\4\5\6\7\8"),
                       (r"^\n",
                        r""),
                       (r"^(=== NEW PHASE ===)|"                        # \1
                        r"^([A-Z]\d{4}[A-Z])",                          # \2
                        r"# \1\2"),
                       (r"^(\d+)(request:.*)",                          # \1 \2
                        r"\1,\2"),
                       (r"^(\d+)(response:.*)|"                         # \1 \2
                        r"^(\d+)(notification:.*)",                     # \3 \4
                        r"\1\3,,\2\4"),
                       (r"(,)[a-z]+:\[(.*?)\]",                         # \1 \2
                        r"\1\2")
                       ]

    with open(filename, "r") as file:
        content = file.read()

    for search_replace in search_replaces:
        content = re.sub(search_replace[0], search_replace[1], content, flags=re.MULTILINE)

    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract DAIDE requests, responses "
                                                 "and notifications from a 1 DAIDE vs 6 others",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("results", metavar="RESULTS_FILE",
                        help="path to the raw results file")
    parser.add_argument("-o", "--output-csv", default=None,
                        help="The csv to save")
    args = parser.parse_args()

    csv_content = daide_script_to_csv(args.results)
    if args.output_csv:
        with open(args.output_csv, "w") as csv_file:
            csv_file.write(csv_content)
    else:
        print(csv_content)
