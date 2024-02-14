"""
Retrieve the original graphs from the TUDataset and save them as graphml.
The corresponding classes are saved in a separate file.
"""
import argparse

from utils import load_graphs_from_TUDataset, save_graphs, set_global_verbose, SAVING_FORMAT


def main(args):
    set_global_verbose(args.verbose)

    nx_graphs, graph_classes = load_graphs_from_TUDataset(args.root_dataset,
                                                          args.dataset)

    save_graphs(args.folder_results, nx_graphs, graph_classes, args.graph_format)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Create Baseline Graphs.\n'
                    '1. Retrieve the graph dataset from the TUDataset repo.\n'
                    '2. Retrieve the classes of the graphs.\n'
                    '3. Transform the graphs from the PyG representation to NetworkX.Graph.\n'
                    '4. Save the graphs and the corresponding classes.\n\n'
                    'Result example:\n'
                    '--------\n'
                    'Folder\n'
                    '  |- graph_0.graphml\n'
                    '  |- graph_1.graphml\n'
                    '  |- ....\n'
                    '  |- graph_classes.cxl')
    subparser = args_parser.add_subparsers()

    args_parser.add_argument('--dataset',
                             type=str,
                             required=True,
                             help='Graph dataset to retrieve'
                                  '(the chosen dataset has to be in the TUDataset repository)')
    args_parser.add_argument('--root-dataset',
                             type=str,
                             default='/tmp/data',
                             help='Root of the TUDataset')

    args_parser.add_argument('--graph-format',
                            type=str,
                            choices=SAVING_FORMAT,
                            default='pkl',
                            help='Format to save the reduced graphs')

    args_parser.add_argument('--folder-results',
                             type=str,
                             required=True,
                             help='Folder where to save the `graphml` graphs and their corresponding classes.')

    args_parser.add_argument('-v',
                             '--verbose',
                             action='store_true',
                             help='Activate verbose print')

    parse_args = args_parser.parse_args()

    main(parse_args)
