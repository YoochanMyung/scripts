# Feature Analysis
- `./Arepggion_analysis`
--`getArpeggio.py`: This is a script for running Arpeggio in parallel. For this, you have to provide paratope and interface information in text which can be obtained by `../getIntRes.py`
--`analyzeArpeggio.py`: It extract Arpeggio interactions in the aspect of CDR and interface residues. It requires a specific format of input TSV file.
--`normaliseArpeggio.py`: This is a script for getting normalised Arpeggio interactions based on BSA information. It requires BSA information as csv.

- `./BSA_analysis`
-- `getBSA.py`: This is a script for calculating Buried Surface Area using FreeSASA.
- `Concavity_analysis`
-- `getRinaccess_ghecom.py`: This script generates a file for running Ghecom. So, users need to run the outcome of this script to get Ghecom results (Rinaccess values).
-- `mapRinaccess_on_bfactor.py`: This script maps Rinaccess values on a given PDB structure.
- `./DSSP_analysis`
-- `mapDssp_bfacator.py`: This script maps DSSP values on a given PDB structure.

# Others
- `cleaenPDB_landscape.py`: It cleans PDB files by removing disordered atoms, alternative conformation other than A, water and ion molecules, and replacing the alternative conformation 'A' to ''.
- `cleanPDB_SabDab.py`: This was designed for Landscapes project. It accepts a specific type of data file such as SAbDab TSV file. The functionality is same as `cleanPDB_landscape.py`.
- `getIntRes.py`: Extract the information of interface residues using Pymol.
