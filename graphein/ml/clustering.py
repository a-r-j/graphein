"""
Utilities for splitting a protein dataset based on sequence homology.

NB. These functions require Blast+ for proper execution. On Linux, this can be installed with:

sudo apt install ncbi-blast+

Otherwise, please see: https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download
"""

import contextlib
import os
import random
import sys
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from Bio import SeqIO


def build_fasta_file_from_mapping(
    pdb_sequence_mapping: Dict[str, str],
    fasta_out: str,
):
    """
    Builds a fasta database from a mapping of PDBs (or other identifier) to sequences.

    :param pdb_sequence_mapping: dictionary with the sequence of each pdb file
    :type pdb_sequence_mapping: Dict[str, str]
    :param fasta_out: file name of the fasta database
    :type fasta_out: str
    """
    with open(fasta_out, mode="w") as f_out:
        for k, v in pdb_sequence_mapping.items():
            seq_id = k
            print(f">{seq_id}\n{v}", file=f_out)


def build_fasta_file_from_graphs(
    graphs: List[nx.Graph], fasta_out: str, chains: Optional[List[str]] = None
):
    if chains is None:
        chains = ["A"] * len(graphs)
    mapping = {
        f"{g.name}_{chain}": g.graph[f"sequence_{chain}"]
        for g, chain in zip(graphs, chains)
    }

    build_fasta_file_from_mapping(mapping, fasta_out)


def get_seq_records(
    filename,
    alphabet=None,  # alphabet=generic_protein,
    check_sequences=False,
    file_format="fasta",
    return_as_dictionary=False,
):
    """
    Get the sequence records from a file.

    :param filename: File with sequences [fasta in our case].
    :type filename: str
    :param alphabet: alphabet to be used.
    :param check_sequences: Sanity check on the sequences.
    :type check_sequences: bool
    :param file_format: We use fasta format (don't change this).
    :type file_format: str
    :param return_as_dictionary: if ``True`` uses ``records.to_dict()`` to convert the record into a dictionary and return it.
    :type return_as_dictionary: bool
    :return: a list of the ``SeqRecord`` of the sequences.
             - ``.seq``: access the records (sequence, i.e. ``Seq``, for actual string use ``str(records[].seq))``
             - ``.id``: access the ids in the file
             - ``.name``: access names (sometimes equal to ``id``, depends on format)
             - ``.description``: is the description (when included)
             - ``.dbxref`` which (if present) is a list with the cross-referenced database
    Notes:
    ------
    For a list of available file_format see http://biopython.org/wiki/SeqIO#File_Formats .aln is generally 'clustal'
    """
    if alphabet is None and check_sequences:
        sys.stderr.write(
            f"WARNING in {get_seq_records.__name__} ask to check_sequences but no "
            "Alphabet given. Only checking for terminating *!\n"
        )
        check_sequences = False
    with open(filename, "r") as handle:
        records = list(SeqIO.parse(handle, file_format, alphabet=alphabet))
    del handle
    if check_sequences:
        for record in records:
            if record.seq[-1] == "*":
                record.seq = record.seq[:-1]
                if alphabet is not None:
                    good = False
                    with contextlib.suppress(ValueError):
                        good = _verify_alphabet(record.seq)
                    if not good:
                        sys.stderr.write(
                            f"WARNING in {get_seq_records.__name__} sequence {record.seq.id} from file "
                            f"{filename} is not compatible with declared alphabet {str(alphabet)}\n"
                        )
    return SeqIO.to_dict(records) if return_as_dictionary else records


def create_pairs_for_clustering(
    fasta_file: str,
    seq_id_low_thresh: float = 25.0,
    use_very_loose_condition: bool = False,
    out_file: Optional[str] = None,
    n_cpu: int = 1,
    max_target_seqs: int = 200,
    delete_blast_databases_when_done: bool = False,
):
    """
    Create sequences pairs for clustering [sequence names MUST be the ids!].

    :param fasta_file: file with fastas
    :type fasta_file: str
    :param seq_id_low_thresh: sequence identity of lower threshold
    :type seq_id_low_thresh: float
    :param use_very_loose_condition: ??
    :type use_very_loose_condition: bool
    :param out_file: name of the output file
    :type out_file: str
    :param n_cpu: number of cpus to be used
    :type n_cpu: int
    :param max_target_seqs: ??
    :type max_target_seqs: int
    :param delete_blast_databases_when_done: delete the db once the process is finished
    :type delete_blast_databases_when_done: bool
    :return:
    """
    if out_file is None:
        out_file = "pairs_for_clustering.txt"
    rec = get_seq_records(fasta_file)
    # blast each sequence on the database to get the identity pairs
    pairs = []
    corresponding_identities = []
    for s in rec:
        pairs += [(s.id, s.id)]
        corresponding_identities += [100.0]
        b_res = blast(
            s,
            fasta_file,
            max_target_seqs=max_target_seqs,
            n_cpu=n_cpu,
            delete_blast_file=True,
        )
        if (
            b_res is None
        ):  # blast sometimes does not hit the very same sequence (with blst+ it should)
            continue
        if use_very_loose_condition:
            _, d_res = process_blast_results(
                b_res,
                len(s.seq),
                return_dictionary_of_numpy_matrices=True,
                filter_best_match_for_same_subject=False,
                add_identity_on_shortest=False,
            )
            for k in d_res:
                if k == s.id:
                    continue  # we added this manually at the beginning
                max_id = max(
                    d_res[k][:, 0]
                )  # maximum identity between different possible alignments
                if not 0 <= max_id <= 100:
                    print(f"Max_id problem: {max_id}, {k}, {s.id}")
                if (
                    max_id > seq_id_low_thresh
                    and frozenset([k, s.id]) not in pairs
                ):
                    pairs += [frozenset([s.id, k])]
                    corresponding_identities += [max_id]
        else:
            b_res = process_blast_results(
                b_res,
                len(s.seq),
                return_dictionary_of_numpy_matrices=False,
                filter_best_match_for_same_subject=True,
                add_identity_on_shortest=True,
            )
            for p in b_res:
                if p[0] == s.id:
                    continue  # we added this manually at the beginning
                if p[-1] > seq_id_low_thresh:
                    pairs += [frozenset([s.id, p[0]])]
                    corresponding_identities += [p[-1]]

    if out_file is not None:
        out = open(out_file, "w")
    done = {}
    for j, p in enumerate(pairs):
        if out_file is not None:
            if p in done:
                continue
            ff = list(p)
            out.write(
                "%s\t%s\t%lf\n" % (ff[0], ff[1], corresponding_identities[j])
            )
        done[p] = corresponding_identities[j]
    if out_file is not None:
        out.close()
    if delete_blast_databases_when_done:
        os.system(f"rm -f {fasta_file}.phr {fasta_file}.pin {fasta_file}.psq")
    return done


def clustering_from_pairs(pairs, out_file=None):
    """
    Clustering of a set of elements given a list of 'homologous' with possibly false negatives
    :param pairs: pair of sequences
    :param out_file: output file
    :return: the clusters
    """
    clusters = []
    j = 0
    key_to_cluster = {}
    for p in pairs:
        p = list(p)
        if p[0] in key_to_cluster and p[1] not in key_to_cluster:
            if p[1] not in clusters[key_to_cluster[p[0]]]:
                clusters[key_to_cluster[p[0]]].append(p[1])
            key_to_cluster[p[1]] = key_to_cluster[p[0]]
        elif p[1] in key_to_cluster and p[0] not in key_to_cluster:
            if p[0] not in clusters[key_to_cluster[p[1]]]:
                clusters[key_to_cluster[p[1]]].append(p[0])
            key_to_cluster[p[0]] = key_to_cluster[p[1]]
        elif p[1] in key_to_cluster:
            if key_to_cluster[p[1]] != key_to_cluster[p[0]]:
                # we leave p1 empty and we put all in p0, loop backward to remove stuff...
                del_id = key_to_cluster[p[1]]
                c = len(clusters[del_id]) - 1
                while c >= 0:
                    el = clusters[del_id][c]
                    if el in key_to_cluster:
                        del key_to_cluster[el]
                    if el not in clusters[key_to_cluster[p[0]]]:
                        clusters[key_to_cluster[p[0]]].append(el)
                        key_to_cluster[el] = key_to_cluster[p[0]]
                    del clusters[del_id][c]
                    c -= 1
        else:  # add a new cluster
            key_to_cluster[p[0]] = j
            if p[0] == p[1]:
                clusters.append([p[0]])
            else:
                clusters.append([p[0], p[1]])
                key_to_cluster[p[1]] = j
            j += 1
    del key_to_cluster
    c = len(clusters) - 1
    while c >= 0:
        if not clusters[c]:
            del clusters[c]
        c -= 1
    if out_file is not None:
        with open(out_file, "w") as out:
            for c in clusters:
                out.write("%s\n" % (" ".join(c)))
    return clusters


def blast(
    sequence,
    database,
    ungapped=False,
    add_sseq=False,
    add_qseq=False,
    tmp_folder="/tmp/",
    blast_path="",
    sequence_name="",
    blast_filename="blast" + str(os.getpid()) + ".txt",
    max_target_seqs=100,
    n_cpu=1,
    delete_blast_file=True,
):
    """
    Given a sequence and a database file (e.g. a .fasta file with some sequences) blasts the sequence on the database

    :param sequence: sequence of reference
    :param database: database against which to blast the sequence
    :param return_csv_data_class: Returns a dict with as keys subject_id and as value the list (without subject_id).
        This also contain a header under .hd
    :param ungapped:
    :param add_sseq: append to the returned list the keywords sseq (aligned part of subject sequence)
    :param add_qseq: append to the returned list the keywords qseq (aligned part of query sequence)
    :param tmp_folder: temporary folder
    :param blast_path: path to blast installation folder
    :param sequence_name: name of the sequence
    :param blast_filename: filename for the blast file
    :param max_target_seqs: maximum number of target sequences
    :param n_cpu: number of cpus
    :param delete_blast_file: if True deletes the blast file at the end
    :return: a list of lists.
             - each list is a possible match (best is at index 0)
             - if there is no matching it returns None
             - The inner part is:
             [subject_id (name of the match), identity (percentage), alignment_length,
             mismatches (number within alignment length), gaps (number within alignment length),
             q_start (start index in the query sequence), q_end (end index),
             s_start (start index in the matching sequence), s_end (end index), evalue,
             nident (number of identical amino acids in aligned region), slen (sublect length)]
    Notes
    -----
    Check that NCBI blast is correctly installed in the folder blast_path
    """

    if tmp_folder is None:
        tmp_folder = ""
    elif tmp_folder != "":
        if tmp_folder[-1] != "/":
            tmp_folder += "/"
        if delete_blast_file and tmp_folder not in blast_filename:
            blast_filename = tmp_folder + blast_filename
    if type(sequence) is not str:  # we assume is a Seq object
        sequence_name = sequence.id
        sequence = str(sequence.seq)
    if blast_path != "" and not os.path.isfile(
        blast_path + "makeblastdb"
    ):  # check if the BLAST_PATH is correct
        raise IOError(
            f"***ERROR*** in run_blast path {blast_path} doesnt lead to blast directory "
            "where makeblastdb should be located"
        )
    seq_file = open(tmp_folder + "sequence_temp.txt", "w")
    seq_file.write(">" + sequence_name + "\n" + sequence + "\n")
    seq_file.close()

    # check if the blast database has already been built, if not build it
    if not (
        os.path.isfile(database + ".phr")
        and os.path.isfile(database + ".pin")
        and os.path.isfile(database + ".psq")
    ):
        try:
            os.system(blast_path + "makeblastdb -dbtype prot -in " + database)
        except Exception:
            print(
                "***ERROR*** in run_blast() cannot build blast database maybe you wish to "
                "set BLAST_PATH to correct directory"
            )
            raise
    ungapped = " -comp_based_stats F  -ungapped" if ungapped else ""
    table_k = "6 sseqid pident length mismatch gapopen qstart qend sstart send evalue nident slen"
    if add_sseq:
        table_k += " sseq"
    if add_qseq:
        table_k += " qseq"

    # SHORT SEQUENCES: -task 'blastp-short' -word_size 2 -seg 'no' -evalue 20000
    if len(sequence) <= 15:
        os.system(
            blast_path
            + "blastp -query "
            + tmp_folder
            + "sequence_temp.txt -db "
            + database
            + " -matrix PAM30 -task 'blastp-short' -word_size 2 -seg 'no' -evalue 20000 -out "
            + blast_filename
            + f" -outfmt '{table_k}'"
            + ungapped
            + " -max_target_seqs "
            + str(max_target_seqs)
            + " -num_threads "
            + str(n_cpu)
        )
    else:
        os.system(
            blast_path
            + "blastp -query "
            + tmp_folder
            + "sequence_temp.txt -db "
            + database
            + " -out "
            + blast_filename
            + f" -outfmt '{table_k}'"
            + ungapped
            + " -max_target_seqs "
            + str(max_target_seqs)
            + " -num_threads "
            + str(n_cpu)
        )
    resultf = open(blast_filename, mode="r").read().splitlines()

    results = []
    if len(resultf) == 0:
        return None
    for line in resultf:
        tmp = line.split("\t")
        numb_list = [tmp[0]] + list(map(lambda x: float(x), tmp[1:]))
        results.append(numb_list)
    if delete_blast_file:
        os.system("rm -f " + blast_filename)
    os.system("rm -f " + tmp_folder + "sequence_temp.txt")
    del resultf
    return results


def process_blast_results(
    blast_results_list,
    len_of_query,
    return_dictionary_of_numpy_matrices=False,
    filter_best_match_for_same_subject=True,
    add_identity_on_shortest=True,
):
    """
    If filter_best_match_for_same_subject is True than only the first among the possible alignments of the query
    with that subject sequence is kept. Blast results are sorted according to subject sequences and then to
    evalue
    :param blast_results_list:
    :param len_of_query:
    :param return_dictionary_of_numpy_matrices:
    :param filter_best_match_for_same_subject:
    :param add_identity_on_shortest:
    :return:
    """
    if not isinstance(len_of_query, int):
        print(
            f"WARNING in {process_blast_results.__name__} type(len_of_query) is not Int, "
            "probably screwing results up!!"
        )
    if blast_results_list is None:
        return ([], {}) if return_dictionary_of_numpy_matrices else []
    if (
        not filter_best_match_for_same_subject
        and not add_identity_on_shortest
        and not return_dictionary_of_numpy_matrices
    ):
        return blast_results_list
    last_subj = ""
    new_results = []
    if return_dictionary_of_numpy_matrices:
        subjects_dict = {}
    for res in blast_results_list:
        if return_dictionary_of_numpy_matrices:
            if res[0] not in subjects_dict:
                subjects_dict[res[0]] = np.array(
                    res[1:], dtype=np.float64, ndmin=2
                )  # this removes the ID
            else:
                subjects_dict[res[0]] = np.vstack(
                    (
                        subjects_dict[res[0]],
                        np.array(res[1:], dtype=np.float64),
                    )
                )
        if filter_best_match_for_same_subject and list(res)[0] == last_subj:
            continue
        if add_identity_on_shortest:
            den = 1.0 * min([res[-1], len_of_query])
            new_results.append(res + [100.0 * res[-2] / den])
        else:
            new_results.append(res)
        last_subj = res[0]
    if return_dictionary_of_numpy_matrices:
        return new_results, subjects_dict
    return new_results


def generate_random_sets(
    clusters,
    number_of_sets=10,
    fraction_in_test=0.1,
    train_set_key="LR",
    test_set_key="TS",
    early_break=True,
):
    """
    Generate random sets from clusters.

    :param clusters: input clusters
    :param number_of_sets: total number of sets to generate
    :param fraction_in_test: fraction of sequences to be put in test
    :param train_set_key: Prefix to add to train set
    :param test_set_key: Prefix to add to test set
    :param early_break: if True, break after a suitable combination of training-test set is generated to feed the model
    :return:
    """
    if type(clusters) is str and os.path.isfile(clusters):
        clusters = open(clusters, mode="r").read().splitlines()
        clusters = [cl.split() for cl in clusters]
    num_ids = 0
    c_sizes = []
    for c in clusters:
        c_sizes += [len(c)]
        num_ids += len(c)
    print(f"{num_ids} ids in {len(clusters)} clusters")
    random.seed(a=42)  # for reproducibility
    n = 0
    in_other_tests = []
    while n < number_of_sets:
        train_set_name = f"{train_set_key}_{n:02}"
        test_set_name = f"{test_set_key}_{n:02}"
        with open(train_set_name, mode="w") as train:
            with open(test_set_name, mode="w") as test:
                ids_in_test = []
                labs_in_test = []
                test_rel_size = 0.0
                seq_in_test = 0
                while test_rel_size < fraction_in_test:
                    cl_id = int(random.random() * len(clusters))
                    while cl_id in in_other_tests:
                        cl_id = int(random.random() * len(clusters))
                    in_other_tests += [cl_id]
                    if len(in_other_tests) == len(clusters):
                        print(
                            f"\nWARNING in {generate_random_sets.__name__} we have already generated test files"
                            " with all clusters.\nNew test files will share clusters with existing ones...\n"
                            f"Starting from {n} included...\n"
                        )
                        in_other_tests = []
                    ids_in_test += [cl_id]
                    for idt in clusters[cl_id]:
                        print(f"{str(idt)}", file=test)
                        labs_in_test.append(idt)
                    test_rel_size += 1.0 * c_sizes[cl_id] / float(num_ids)
                    seq_in_test += c_sizes[cl_id]
                for j in range(len(clusters)):
                    if j not in ids_in_test:
                        for idt in clusters[j]:
                            print(f"{str(idt)}", file=train)
                print(
                    f"Generated set {train_set_name} and {test_set_name} - {seq_in_test} sequences"
                    f" for {len(ids_in_test)} clusters in test, among ({num_ids}, {len(clusters)}) in total."
                    f" Test relative size is {test_rel_size}"
                )
        if abs(test_rel_size - fraction_in_test) < 0.05 and early_break:
            return labs_in_test
        n += 1


def train_and_test_from_fasta(
    fasta_file: str,
    number_of_sets: int = 10,
    fraction_in_test: float = 0.1,
    cluster_file_name: str = "s2d_clusters.txt",
    seq_id_low_thresh: float = 25.0,
    use_very_loose_condition: bool = False,
    n_cpu: int = 2,
    max_target_seqs: int = 200,
    delete_db_when_done: bool = True,
    train_set_key: str = "LR",
    test_set_key: str = "TS",
    early_break: bool = True,
):
    """
    Use a fasta database to generate test and training sets based on user defined parameters.

    :param fasta_file: Fasta file name.
    :type fasta_file: str
    :param number_of_sets: Number of sets to generate.
    :type number_of_sets: int
    :param fraction_in_test: Fraction of proteins to be put in test set.
    :type fraction_in_test: float
    :param cluster_file_name: Name of the cluster file.
    :type cluster_file_name: str
    :param seq_id_low_thresh: Sequence identity lower threshold.
    :type seq_id_low_thresh: float
    :param use_very_loose_condition: ignore
    :type use_very_loose_condition: bool
    :param n_cpu: Number of cpus.
    :type n_cpu: int
    :param max_target_seqs: Maximum number of target sequences.
    :type max_target_seqs: int
    :param delete_db_when_done: if ``True`` delete the blast database when done.
    :type delete_db_when_done: bool
    :param train_set_key: Prefix to append to training files.
    :type train_set_key: str
    :param test_set_key: Prefix to append to test files.
    :type test_set_key: str
    :param early_break: if ``True``, break after a suitable combination of training-test set is generated to feed the model.
    :type early_break: bool
    :return: save clusters in files (or return the 1st suitable one).
    """
    if not 0 < fraction_in_test < 1:
        print(
            f"ERROR in {train_and_test_from_fasta.__name__}, fraction_in_test has to be in [0,1] now {fraction_in_test}"
        )

    print(
        "Clustering proteins by sequence identity\n"
        "----------------------------------------\n"
        "Parameters:\n"
        f"    - Number of sets to split: {number_of_sets}\n"
        f"    - Sequence identity low threshold: {seq_id_low_thresh}%\n"
        f"    - Fraction of sequences in test set: {fraction_in_test}"
    )
    print("\n\n*** Creating sequences pairs for clustering")
    pairs = create_pairs_for_clustering(
        fasta_file,
        seq_id_low_thresh=seq_id_low_thresh,
        use_very_loose_condition=use_very_loose_condition,
        out_file=None,
        n_cpu=n_cpu,
        max_target_seqs=max_target_seqs,
        delete_blast_databases_when_done=delete_db_when_done,
    )
    print("*** Clustering from pairs of sequences")
    clusters = clustering_from_pairs(pairs, out_file=cluster_file_name)
    print("*** Generating sets")
    if early_break:
        return generate_random_sets(
            clusters,
            number_of_sets=number_of_sets,
            fraction_in_test=fraction_in_test,
            train_set_key=train_set_key,
            test_set_key=test_set_key,
            early_break=early_break,
        )

    generate_random_sets(
        clusters,
        number_of_sets=number_of_sets,
        fraction_in_test=fraction_in_test,
        train_set_key=train_set_key,
        test_set_key=test_set_key,
        early_break=early_break,
    )
    return
