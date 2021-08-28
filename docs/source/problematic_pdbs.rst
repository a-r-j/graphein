======================
Problematic PDB Files
======================

We include here an excellent overview of common problems in PDB files by Eric Martz. The original text is available here <https://www.umass.edu/microbio/chime/pe_beta/pe/protexpl/badpdbs.htm>_

Most published PDB files are handled correctly by PE, including those with sequence insertions or sequence microheterogeneity. (For further explanation and examples, see the help on Sequence Irregularities, also available from within the Sequences and Seq3D displays.)
Disulfide bonds in multiple-model PDB files.
A bug in Chime, makes it fail to separate disulfide bonds according to model in multiple-model PDB files, typically resulting from NMR experiments. The consequence is that when all models are displayed (in PE's expert mode, or in the NMR Models/Animation control panel), spurious disulfide bonds are drawn between different models. An example is 1AS5.

Alternate Locations (Rotamers).
Many, perhaps the majority, of PDB files have coordinates for alternate locations (rotamers) for a subset of the sidechains. Chime has no mechanism to distinguish these, and simply displays both rotamers on top of each other. Further, since Chime assigns covalent bond positions based on interatomic distances, it typically shows spurious bonds between atoms in the two rotamer positions. For example, 1CBN contains alternate rotamer positions for Thr1, Thr2, Ile7, Val8, Arg10, and so forth. In rare cases, even the alpha carbon atoms also have slightly different positions (Thr1 in 1CBN). Visualization programs that handles alternate locations much better include Cn3D and Jmol.

Sequence Irregularities: The following PDB files, published in the Protein Data Bank, have sequence irregularities that are handled incorrectly by PE.

When inserted residues lack insertion letters (probably illegal according to the PDB format specification), PE fails to show the inserted residues in Sequences/Seq3D. For example, the single unnamed chain in 1DPO contains insertions at postions 184 (Gly, Phe), 188 (Gly, Lys), and 221 (Ala, Leu) but no insertion letters. Clicking on Gly184 in Seq3D highlights both Gly184 and Phe184.

When insertions have more than one copy of the same amino acid (or nucleotide) in a single insertion block, clicking one copy in Seq3D highlights all copies within the same insertion block. For example, chain B in 1IGY contains a block of four residues inserted at sequence position 82. The block contains Leu-Ser-Ser-Leu. Clicking on either Leu highlights both of them in the molecular view, and similarly for either Ser. This results from the inability of Chime to use insertion codes in selecting atoms. For example, Chime cannot distinguish Leu82 from Leu82C.

PE always labels amino acids within an inserted block in alphabetic order, starting with A, in PE's Sequences and Seq3D, regardless of how they are labeled in the ATOM records of the PDB file. For example, chain E in 1HAG begins with 1H, 1G, 1F, ... 1A, then 1 (in reverse alphabetic order). PE labels the inserted block 1, 1A, 1B, 1C, ... 1H (in alphabetic order). This is due to design flaws in PE. Because these cases are rare, the effort to remedy these flaws doesn't seem worthwhile.

Sequence numbers in decreasing order within a single chain cause PE's Sequences report to contain errors, and cause Seq3D to fail to work properly. This is due to design flaws in PE. Because these cases are rare, the effort to remedy these flaws doesn't seem worthwhile. An example is 1NSA, which contains a single (unnamed) protein chain with sequence 7A-95A that continues 4-308. Another example is 1IAO, which contains in chain B (in this order) 1S, 323P-334P, 6-94, 94A, 95-188, 1T, 2T. PE detects such situations and alerts the user that the Sequences/Seq3D displays are garbled.

Chain naming irregularities.

When a single PDB file contains both an unnamed and named chains, PE's mechanisms that depend upon chain names fail. This is probably quite rare, but occurs in 4CPA, where QuickViews's SELECT menu lists the unnamed chain, but clicking on it selects all chains. Also, in Seq3D, clicking on D16 in the unnamed chain highlights D16 there and in chain I (which also happens to have D at position 16).

For reasons not apparent (by inspection of the PDB file), Chime's "show sequence" report fails to include the chain name "A" for 1QCQ. This causes PE to think the chain has no name. In ConSurf, this causes PE to fail to apply the conservation colors in Seq3D.
