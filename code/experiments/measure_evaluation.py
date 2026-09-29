"""Run the unchanged evaluator and record the process peak resident memory."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys


def peak_working_set():
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                    'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        read = ctypes.windll.psapi.GetProcessMemoryInfo
        read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not read(process(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize
    import resource
    amount = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return amount if sys.platform == 'darwin' else amount*1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.argv = [str(root/'evaluate.py'), '--checkpoint', str(args.checkpoint),
                '--device', 'cpu', '--precision', 'fp32', '--threads', str(args.threads),
                '--split', args.split, '--output', str(args.output)]
    runpy.run_path(str(root/'evaluate.py'), run_name='__main__')
    peak = peak_working_set()
    result = {'peak_working_set_bytes': peak, 'peak_working_set_gib': peak/2**30,
              'checkpoint_bytes': args.checkpoint.stat().st_size,
              'measurement': 'process lifetime peak including loading and scoring'}
    output = args.output.with_suffix('.resources.json')
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
