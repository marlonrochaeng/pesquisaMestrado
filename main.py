import argparse, sys
from orquestrador import Orquestrador
import time



parser=argparse.ArgumentParser()

parser.add_argument('--jobs', help='number of jobs')
parser.add_argument('--machines', help='number of jobs to schedule')
parser.add_argument('--path', help='path for the jobs instance')
parser.add_argument('--numInd', help='number of individuals')
parser.add_argument('--numGen', help='number of generations')
parser.add_argument('--toMatrix', help='number of generations')


args=parser.parse_args()

start_time = time.time()
o = Orquestrador(int(args.jobs), int(args.machines), args.path, int(args.numInd), int(args.numGen), float(args.toMatrix))
o.run_eda()
print("--- %s seconds ---" % (time.time() - start_time))
