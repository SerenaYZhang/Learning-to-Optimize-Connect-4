# run full pipeline: games -> retrain -> done

import sys
import os
import subprocess

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Step 1: Running AI vs AI games...")
    try:
        subprocess.run([sys.executable, os.path.join(script_dir, 'ai_vs_ai_games.py')], 
                      cwd=script_dir, check=True)
        print("Games done\n")
    except:
        print("Error running games")
        return
    
    print("Step 2: Retraining model...")
    try:
        subprocess.run([sys.executable, os.path.join(script_dir, 'retrain_with_new_data.py')], 
                      cwd=script_dir, check=True)
        print("Retraining done\n")
    except:
        print("Error retraining")
        return
    
    print("Done! Test in GUI or run metrics.")


if __name__ == "__main__":
    main()

