# -*- coding: utf-8 -*-
import argparse
from demo import async_demo

def main():
    demo_parser = argparse.ArgumentParser()
    demo_parser.add_argument('--ngram_extraction', action='store_true', help="True if the demo is only supposed to process a file for building ngram dictionary")
    demo_parser.add_argument('--ip', type=str, default='127.0.0.1', help="IP Address")
    demo_parser.add_argument('--model_type', type=str, default='', help="model type name (baseline, player_norm, ...)")
    demo_parser.add_argument('--port', type=int, default=6006, help="port number that server will listen to")
    demo_parser.add_argument("--data_dir", default="./data", type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    demo_parser.add_argument('--ngram_distribution', type=str, default="./demo/ngram_distribution.json", help="ngram dictionary file address")
    demo_parser.add_argument('--ngram_source', type=str, default="./ngram_source.txt", help="source file for extracting the ngrams")
    args = demo_parser.parse_args()

    async_demo.main(args)

if __name__ == "__main__":
    main()
