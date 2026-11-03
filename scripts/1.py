"""Monitor/retrieve your saved notebook version; run using your Kaggle identity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
if __name__=='__main__':
    if '--isec-final-evaluation' in sys.argv:
        import runpy
        runpy.run_path(str(Path(__file__).with_name('isec_final_evaluation.py')),run_name='__main__')
    elif '--adjscc-evaluation' in sys.argv:
        import runpy
        runpy.run_path(str(Path(__file__).with_name('adjscc_evaluation.py')),run_name='__main__')
    elif '--campaign' in sys.argv:
        sys.argv.remove('--campaign')
        from src.person1.campaign import main
        main()
    else:
        from src.launch import main
        main('1')
