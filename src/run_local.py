from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from TCA.execute import tca_plus_bellw
from NAIVE.execute import bellw, bellw_local
from pdb import set_trace
from data.handler import get_all_projects


def main():
    project = get_all_projects()['Apache']
    local_result = bellw_local(source=project, target=project, n_rep=12)
    bellw_result = bellw(source=project, target=project, n_rep=12)
    tcap_result = tca_plus_bellw(source=project, target=project, n_rep=12)
    set_trace()


if __name__ == "__main__":
    main()