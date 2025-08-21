"""
Main entry point for GuPPy (Guided Photometry Analysis in Python)
"""
import panel as pn
from .savingInputParameters import savingInputParameters

def main():
    """Main entry point for GuPPy"""
    template = savingInputParameters()
    pn.serve(template, show=True)

if __name__ == "__main__":
    main()