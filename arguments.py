import argparse

prob_cond_metrics = {
        'binary': ['regions'],
#       'binary': ['path-length'],
#       'binary': ['regions', 'path-length'],
        'zelda': ['num_enemies'],
        'sokoban': ['num_boxes'],
        }

def get_args():
    opts = argparse.ArgumentParser(description='Conditional PCGRL')
    opts.add_argument(
        '-p',
        '--problem',
        help='which problem (i.e. game) to generate levels for (binary, sokoban, zelda, mario, ... rct, simcity???)',
        default='binary')
    opts.add_argument(
        '-r',
        '--representation',
        help='Which representation to use (narrow, turtle, wide, ... cellular-automaton???)',
        default='turtle')
    opts.add_argument(
        '-c',
        '--conditionals',
        nargs='+',
        help='Which game level metrics to use as conditionals for the generator',
        default=prob_cond_metrics['binary'])
    opts.add_argument(
        '--resume',
        help='Are we resuming from a saved training run?',
        action="store_true",)
    opts = opts.parse_args()

    return opts
