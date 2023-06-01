import turtle
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import pysam
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
import pyBigWig
from typing import Optional
from collections import Counter
from loguru import logger
import utils


def preprocessing(infiles: list, methylation: list, chrom: str, start: int, end: int, bam: str = None, mapq: int = 0):
    '''For each read, if it appears in all the infiles, then add it to the read_set
    
    Parameters
    ----------
    infiles : list
        list of input files
    methylation: list
        A boolean list. if value is True, only keep the reads with deepsignal results
    chrom : str
        chromosome name
    start : int
        the start position of the region you want to look at
    end : int
        end of the region to be processed
    Returns
    -------
        A set of read_ids that are present in all infiles.
    
    '''

    if len(infiles) != len(methylation):
        raise ValueError("The length of infiles and methylation must be the same.")

    read_counter = Counter()
    for infile, methy in zip(infiles, methylation):
        tbx = pysam.TabixFile(infile)
        for line in tbx.fetch(chrom, start, end):
            read_id, mod, _chrom, _start, _end, strand, methylated_pos, unmethylated_pos = line.split('\t')  # unmethylated_pos only work for 5mC

            if methy:
                unmethylated_pos = np.fromstring(unmethylated_pos, sep=',', dtype=int)
                unmethylated_pos = unmethylated_pos[((unmethylated_pos >= start) & (unmethylated_pos <= end))]

                methylated_pos = np.fromstring(methylated_pos, sep=',', dtype=int)
                methylation_pos = methylated_pos[((methylated_pos >= start) & (methylated_pos <= end))]
                if len(unmethylated_pos) > 0 or len(methylation_pos) > 0:
                    read_counter[read_id] += 1
            else:
                read_counter[read_id] += 1
    
    read_set = set()
    for read_id, count in read_counter.items():
        if count == len(infiles):
            read_set.add(read_id)
    
    if bam is not None:
        bam_read_set = set()
        with pysam.AlignmentFile(bam, 'rb') as bamfile:
            for read in bamfile.fetch(chrom, start, end):
                if read.mapping_quality >= mapq:
                    bam_read_set.add(read.query_name)
        
        read_set = read_set & bam_read_set
    
    return read_set


def get_soft_clip_from_bam(infile: list, chrom: str, start: int, end: int):
    soft_clip = {}
    bamfile = pysam.AlignmentFile(infile, 'rb')
    for read in bamfile.fetch(chrom, start, end):
        left_softclip = read.cigar[0][1] if read.cigar[0][0] == 4 else 0
        right_softclip = read.cigar[-1][1] if read.cigar[-1][0] == 4 else 0
        read_id = read.query_name

        if read_id not in soft_clip:
            soft_clip[read_id] = {
                'left_softclip': left_softclip,
                'right_softclip': right_softclip,
            }
        else:
            logger.warning(f"Duplicate read id {read_id} in bam file.")
    
    return soft_clip


def subsample_reads(mod_results, read_info, fraction: float = 50):
    sample_count = len(mod_results)
    if fraction <= 1:
        subsample_count = int(sample_count * fraction)
    else:
        subsample_count = fraction
    
    if subsample_count >= sample_count:
        logger.warning("The subsample count is larger than the sample count, return the original data.")
        return mod_results, read_info

    np.random.seed(42)
    subsample_index = sorted(np.random.choice(sample_count, subsample_count, replace=False))
    mod_results = np.array(mod_results)[subsample_index, :]
    read_info = (np.array(read_info)[subsample_index, :2]).astype(int)

    return list(mod_results), list(read_info)


def get_gene_model(chrom: str, start: int, end: int, *bed_paths: tuple) -> pd.DataFrame:
    """Get gene model information from the bed file.
        The bed file should be indexed by tabix.
        For details, see http://www.htslib.org/doc/tabix.html

    Args:
        chrom (str): chromosome id.

        start (int): the start of the region.

        end (int): the end of the region.

        bed_path (str): the PATH of the bed file.

    Returns:
        pd.DataFrame: A bed12 dataframe.
            Examples
            --------
            chrom     start       end      gene_id score strand thickStart  thickEnd
                2  17922018  17924542  AT2G43110.1     0      +   17922066  17924291

            rgb blockCount              blockSizes                blockStarts
              0          6  361,196,73,50,114,372,  0,527,823,1802,1952,2152,
    """
    gene_models = None
    for bed_path in bed_paths:
        tbx = pysam.TabixFile(bed_path)
        _gene_model = pd.DataFrame(
            [gene_model.split("\t")[:12] for gene_model in tbx.fetch(chrom, start, end)],
            columns=[
                "chrom", "start", "end", "gene_id", "score", "strand", "thickStart", "thickEnd", "rgb", "blockCount", "blockSizes", "blockStarts"],
        )
        if gene_models is None:
            gene_models = _gene_model
        else:
            gene_models = gene_models.append(_gene_model)

    gene_models.sort_values(["start", "end"], inplace=True)
    gene_models.reset_index(drop=True, inplace=True)

    return gene_models


def get_y_pos_continuous(df, gene_list=None, threshold=0):
    """Get the y position of each region. Save the results in the df['y_pos'].

    Args:
        df (pd.DataFrame): a DataFrame must include first four columns:
            chrom, start, end, gene_id.

            Examples
            --------
            chrom    start      end    gene_id strand
                4  9105672  9106504  AT4G16100      +

        gene_list (set, optional): a set contain which gene to plot.
            When gene_list is None, plot all item in the df. Defaults to None.

        threshold (int, optional): the minimum space between two region. Defaults to 0.

    Returns:
        int: item counts in the y position.
    """
    # initialization of y_pos columns
    df["y_pos"] = None

    read_list = []
    for index_id, item in enumerate(df.values):
        chrom, start, end, gene_id, *_ = item

        # only plot reads/gene_model in the gene_list
        if gene_list is not None and gene_id not in gene_list:
            df.drop(index_id, inplace=True)
            continue

        current_read = (chrom, start, end)
        is_add_read = False
        y_pos = -1
        for y_pos in range(len(read_list)):
            y_pos_read = read_list[y_pos]
            if not is_overlap(current_read, y_pos_read, threshold=threshold):
                read_list[y_pos] = current_read
                df.at[index_id, "y_pos"] = y_pos
                is_add_read = True
                break
        if not is_add_read:
            read_list.append(current_read)
            y_pos += 1
            df.at[index_id, "y_pos"] = y_pos

    return len(read_list)


def is_overlap(
    gene_a, gene_b, threshold: int = 0) -> bool:
    """To judge whether two region is overlap.

    Args:
        gene_a (tuple): (chrom, start, end)
        gene_b (tuple): (chrom, start, end)
        threshold (int, optional): the minimum space between two region. Defaults to 0.

    Returns:
        bool: if overlap True, else False
    """
    minn = max(int(gene_a[1]), int(gene_b[1]))
    maxn = min(int(gene_a[2]), int(gene_b[2]))

    if maxn - minn >= -threshold:
        return True
    else:
        return False


def plot_small_arrow(ax, xpos, ypos, strand, auto_arrow=1):
    """
    Draws a broken line with 2 parts:
    For strand = +:  > For strand = -: <
    :param xpos:
    :param ypos:
    :param strand:
    :
    :return: None
    """
    current_small_relative = 20*auto_arrow
    if strand == '+':
        xdata = [xpos - current_small_relative / 4,
                    xpos + current_small_relative / 4,
                    xpos - current_small_relative / 4]
    elif strand == '-':
        xdata = [xpos + current_small_relative / 4,
                    xpos - current_small_relative / 4,
                    xpos + current_small_relative / 4]
    else:
        return
    ydata = [ypos - 2, ypos, ypos + 2]

    ax.add_line(Line2D(xdata, ydata, linewidth=1, color='w'))


def plot_gene_model(
    ax,
    gene_models: pd.DataFrame,
    fig_start: int,
    fig_end: int,
    gene_color: str = "k",
    y_space: int = 1,
    arrow: bool = True,
    small_arrow: bool = False,
    annotation_pos: str = 'left',
    auto_arrow: int = 1,
):
    """plot gene model in the axis

    Args:
    -----
        ax (matplotlib.axes): An axis object to plot.

        gene_models (pd.DataFrame): A bed12 like DataFrame:

            chrom     start       end      gene_id score strand thickStart  thickEnd
                2  17922018  17924542  AT2G43110.1     0      +   17922066  17924291

            rgb blockCount              blockSizes                blockStarts y_pos
              0          6  361,196,73,50,114,372,  0,527,823,1802,1952,2152,     0

        fig_start (int): figure xlim start.

        fig_end (int): figure xlim end.

        gene_color (str, optional): gene color. Defaults to 'k'.

        y_space (int, optional): the spaces between gene in y direction. Defaults to 1.
    """
    ylim = 0  # ax ylim的下限
    height = 3  # gene model 高度
    y_space = y_space + height * 2
    for gene_model in gene_models.values:
        (
            chrom,
            chromStart,
            chromEnd,
            gene_id,
            _,
            strand,
            thickStart,
            thickEnd,
            _,
            blockCount,
            blockSizes,
            blockStarts,
            y_pos,
        ) = gene_model
        y_pos = -y_space * y_pos
        ylim = min(y_pos, ylim)

        # 数据类型转化
        chromStart = int(chromStart)
        chromEnd = int(chromEnd)
        thickStart = int(thickStart)
        thickEnd = int(thickEnd)
        blockSizes = np.fromstring(blockSizes, sep=",", dtype="int")
        blockStarts = np.fromstring(blockStarts, sep=",", dtype="int") + chromStart

        # 画转录起始位点及方向箭头
        if arrow:
            small_relative = 0.1 * (chromEnd - chromStart)  # 箭头突出部分相对长度
            arrowprops = dict(arrowstyle="-|>", connectionstyle="angle", color=gene_color)
            if strand == "+":
                ax.annotate(
                    "",
                    xy=(chromStart + small_relative, height * 2.5 + y_pos),
                    xytext=(chromStart, y_pos),
                    arrowprops=arrowprops,
                )
            else:
                ax.annotate(
                    "",
                    xy=(chromEnd - small_relative, height * 2.5 + y_pos),
                    xytext=(chromEnd, y_pos),
                    arrowprops=arrowprops,
                )

        line = mp.Rectangle(
            (chromStart, y_pos - height / 8),
            chromEnd - chromStart,
            height / 4,
            color=gene_color,
            linewidth=0,
        )  # 基因有多长这条线就有多长
        ax.add_patch(line)

        for exonstart, size in zip(blockStarts, blockSizes):
            if exonstart == chromStart and exonstart + size == chromEnd:
                utr_size = thickStart - chromStart
                utr = mp.Rectangle(
                    (exonstart, y_pos - height / 2),
                    utr_size,
                    height,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(utr)
                utr_size = chromEnd - thickEnd
                utr = mp.Rectangle(
                    (thickEnd, y_pos - height / 2),
                    utr_size,
                    height,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(utr)
                exon = mp.Rectangle(
                    (thickStart, y_pos - height),
                    thickEnd - thickStart,
                    height * 2,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(exon)

            elif exonstart + size <= thickStart:
                # 只有5'/ 3'UTR
                utr = mp.Rectangle(
                    (exonstart, y_pos - height / 2),
                    size,
                    height,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(utr)

            elif exonstart < thickStart and exonstart + size > thickStart:
                # 带有5' / 3' UTR的exon
                utr_size = thickStart - exonstart
                utr = mp.Rectangle(
                    (exonstart, y_pos - height / 2),
                    utr_size,
                    height,
                    color=gene_color,
                    linewidth=0,
                )
                exon = mp.Rectangle(
                    (exonstart + utr_size, y_pos - height),
                    size - utr_size,
                    height * 2,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(utr)
                ax.add_patch(exon)

            elif exonstart >= thickStart and exonstart + size <= thickEnd:
                # 普通exon
                exon = mp.Rectangle(
                    (exonstart, y_pos - height),
                    size,
                    height * 2,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(exon)

            elif exonstart < thickEnd and exonstart + size > thickEnd:
                # 带有3' / 5' UTR的exon
                utr_size = exonstart + size - thickEnd
                utr = mp.Rectangle(
                    (thickEnd, y_pos - height / 2),
                    utr_size,
                    height,
                    color=gene_color,
                    linewidth=0,
                )
                exon = mp.Rectangle(
                    (exonstart, y_pos - height),
                    size - utr_size,
                    height * 2,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(utr)
                ax.add_patch(exon)

            elif exonstart >= thickEnd:
                # 只有3'/ 5'UTR
                utr = mp.Rectangle(
                    (exonstart, y_pos - height / 2),
                    size,
                    height,
                    color=gene_color,
                    linewidth=0,
                )
                ax.add_patch(utr)

        if annotation_pos == "top":
            ax.annotate(
                gene_id, xy=((chromStart + chromEnd) / 2, y_pos + y_space), ha="center",
            )
        elif annotation_pos == "bottom":
            ax.annotate(
                gene_id, xy=((chromStart + chromEnd) / 2, y_pos - y_space), ha="center",
            )
        elif annotation_pos == 'topleft':
            ax.annotate(
                gene_id, xy=(chromStart, y_pos + height * 1.5), ha="left", fontsize=8
            )
        elif annotation_pos == "left":
            ax.annotate(
                gene_id, xy=((chromStart-(fig_end-fig_start)*.01), y_pos), ha="right",
                va='center'
            )
        else:
            pass
            
        # TODO: arrow color, arrow current_small_relative
        if small_arrow:
            # pos = np.linspace(chromStart, chromEnd, 4)[1:-1]
            pos = np.arange(chromStart, chromEnd, 55*auto_arrow)[1:-1]
            for x_pos in pos:
                plot_small_arrow(ax, x_pos, y_pos, strand, auto_arrow=auto_arrow)


    # set ax
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_ticks_position("none")

    ax.set_ylim(ylim - height - y_space, height)
    ax.set_xlim(fig_start, fig_end)


def filter_low_cov(bw_data, bw_all, min_cov: int = 10):
    boo = bw_all < 10
    bw_data[boo] = .01


def get_mod_results(infile: str, chrom: str, start: int, end: int, fully_span: bool = False, left_span: bool = False, right_span: bool = False, start_before: int = None, end_after: int = None, expand: int = 5, unmethylated: bool = False, filter_read: bool = False, read_set: set = None, add_bigwig:bool = False, stranded: bool = False, soft_clip: dict = None):
    '''This function takes a bed file and returns a list of the lines in the file that overlap with the
    specified region
    
    Parameters
    ----------
    infile : str
        the modification results. bgziped and tabix indexed are required
    chrom : str
        chromosome
    start : int
        the start position of the region you want to get the results for
    end : int
        int = end of the region you want to get the results for
    fully_span : bool, optional
        if True, will return all results that fully span the region. If False, will return all results that overlap the region.
    left_span:  bool, optional
        if True, will return all results that  span the left region.
    right_span:  bool, optional
        if True, will return all results that  span the right region.
    right_span: bool, optional
    expand : int, optional
        how many bases to expand the region by on either side
    read_set: set, optional
        a set of read names to filter the results by. If None, will not filter by read name
    add_bigwig: bool, optional
        if True, convert the methylation level from mod results
    strand: bool, option
        if True, the methylation status will be shown on the forward and reverse strand separately
    soft_clip: dict, optional
        a dict of soft clip position. If None, will not add soft clip length to the read length
    '''

    tbx = pysam.TabixFile(infile)

    region_length = end - start

    mod_results = []  # read mod_bases position
    read_info = []  # start, end, read_id, strand
    # fow bigwig tracks
    if add_bigwig:
        if stranded:
            bw_methy_plus, bw_unmethy_plus = np.zeros(region_length, dtype=int), np.zeros(region_length, dtype=int)
            bw_methy_minus, bw_unmethy_minus = np.zeros(region_length, dtype=int), np.zeros(region_length, dtype=int)
        else:
            bw_methy, bw_unmethy = np.zeros(region_length, dtype=int), np.zeros(region_length, dtype=int)

    for line in tbx.fetch(chrom, start, end):
        read_id, mod, _chrom, _start, _end, strand, methylated_pos, unmethylated_pos = line.split('\t')

        # filter by read_id
        if read_set is not None and read_id not in read_set:
            continue

        _start = int(_start)
        _end = int(_end)
        methylated_pos = np.fromstring(methylated_pos, sep=',', dtype=int)
        unmethylated_pos = np.fromstring(unmethylated_pos, sep=',', dtype=int)

        if fully_span and not(_start <= start and _end > end):
            continue
        if left_span and not(_start <= start):
            continue
        if right_span and not(_end > end):
            continue
        if start_before is not None and not(_start <= start_before):
            continue
        if end_after is not None and not(_end > end_after):
            continue

        mod_res = np.zeros(region_length, dtype='int8')
            
        _methylated_pos = methylated_pos[((methylated_pos >= start) & (methylated_pos <= end))] - start - 1
        _unmethylated_pos = unmethylated_pos[((unmethylated_pos >= start) & (unmethylated_pos <= end))] - start - 1
        mod_res[_methylated_pos] = 1

        if (_methylated_pos < _start - start - 1).any():
            continue  # filter mapping error

        if unmethylated:
            _unmethylated_pos = unmethylated_pos[((unmethylated_pos >= start) & (unmethylated_pos <= end))] - start - 1
            mod_res[_unmethylated_pos] = -1  # unmethylated labeled as -1

        for i in range(expand):
            _genome_pos1 = _methylated_pos - i
            _genome_pos1 = _genome_pos1[_genome_pos1 > 0]
            mod_res[_genome_pos1] = 1
            _genome_pos2 = _methylated_pos + i
            _genome_pos2 = _genome_pos2[_genome_pos2 < region_length]
            mod_res[_genome_pos2] = 1

        if filter_read and not((mod_res == -1).any() or (mod_res == 1).any()):
            continue  # discard reads with no deepsignal called results

        if add_bigwig:
            if stranded:
                if strand == '+':
                    bw_methy_plus[_methylated_pos] += 1
                    bw_unmethy_plus[_unmethylated_pos] += 1
                else:
                    bw_methy_minus[_methylated_pos] += 1
                    bw_unmethy_minus[_unmethylated_pos] += 1
            else:
                bw_methy[_methylated_pos] += 1
                bw_unmethy[_unmethylated_pos] += 1

        mod_results.append(mod_res)
        read_start = _start - start if _start > start else 0
        read_end = _end - start if end - _end > 0 else end-start

        # add soft clip length
        if soft_clip is not None:
            if read_id in soft_clip:
                if right_span:
                    read_start = _start - start
                    read_start -= soft_clip[read_id]['left_softclip']
                if left_span:
                    read_end = _end - start
                    read_end += soft_clip[read_id]['right_softclip']

        read_info.append((read_start, read_end, read_id, strand))  # start, end, read_id, strand

    if add_bigwig:
        if stranded:
            bw_data_plus = bw_methy_plus / (bw_unmethy_plus+bw_methy_plus)
            bw_data_minus = bw_methy_minus / (bw_unmethy_minus+bw_methy_minus)
            bw_data_plus = np.nan_to_num(bw_data_plus)
            bw_data_minus = np.nan_to_num(bw_data_minus)
            filter_low_cov(bw_data_plus, bw_unmethy_plus+bw_methy_plus)
            filter_low_cov(bw_data_minus, bw_unmethy_minus+bw_methy_minus)
            return mod_results, read_info, bw_data_plus, bw_data_minus
        else:
            bw_data = bw_methy / (bw_unmethy+bw_methy)
            bw_data = np.nan_to_num(bw_data)
            filter_low_cov(bw_data, bw_unmethy+bw_methy)
            return mod_results, read_info, bw_data
        
    return mod_results, read_info


def mod_results_stranded(mod_results, read_info):
    mod_results_fwd, read_info_fwd, mod_results_rev, read_info_rev = [], [], [], []
    for mod, read in zip(mod_results, read_info):
        if read[3] == '+':
            mod_results_fwd.append(mod)
            read_info_fwd.append(read)
        elif read[3] == '-':
            mod_results_rev.append(mod)
            read_info_rev.append(read)
        else:
            raise ValueError('strand should be + or -')

    return mod_results_fwd, read_info_fwd, mod_results_rev, read_info_rev


def read_clustering(mod_results: list, read_info: list):
    '''It takes in a list of modified results and a list of read info, and returns a sorted list of
    modified results and a sorted list of read info
    
    Parameters
    ----------
    mod_results : list
        list of lists, each list is a list of the modified positions of a read
    read_info : list
        list of tuples, each tuple is (read_start, read_end, read_id)
    
    Returns
    -------
        mod_results_sorted is a list of lists, each list is a list of the modified results for a read.
        read_info_sorted is a list of lists, each list is a list of the read information for a read.
    
    '''
    Z = linkage(mod_results, method='single', metric='euclidean')
    clusters = fcluster(Z, 0, criterion='distance')  # 属于哪一个cluster
    clusters_sorted_index = np.argsort(clusters)
    mod_results_sorted = [mod_results[i] for i in clusters_sorted_index]
    read_info_sorted = [read_info[i] for i in clusters_sorted_index]

    return mod_results_sorted, read_info_sorted


def read_list_clustering(combine_mod_results: list, mod_results: list, read_info: list):
    Z = linkage(combine_mod_results, method='single', metric='euclidean')
    clusters = fcluster(Z, 0, criterion='distance')  # 属于哪一个cluster
    clusters_sorted_index = np.argsort(clusters)

    mod_results_sorted, read_info_sorted = [], []
    for _mod_results, _read_info in zip(mod_results, read_info):
        mod_results_sorted.append([_mod_results[i] for i in clusters_sorted_index])
        read_info_sorted.append([_read_info[i] for i in clusters_sorted_index])

    return mod_results_sorted, read_info_sorted


def read_list_sorting(mod_results: list, read_info: list, sorted_index):
    mod_results_sorted, read_info_sorted = [], []
    for _mod_results, _read_info in zip(mod_results, read_info):
        mod_results_sorted.append([_mod_results[i] for i in sorted_index])
        read_info_sorted.append([_read_info[i] for i in sorted_index])
    
    return mod_results_sorted, read_info_sorted


def read_sorting_by_id(mod_results: list, read_info: list, read_id_sorted: list):
    '''Use the provided read_id_sorted to sort the mod_results and read_info
    
    Parameters
    ----------
    mod_results : list
        a list of lists, each sublist is a list of the following format:
    read_info : list
        list of lists, each sublist contains the following information:
    read_id_sorted : list
        a list of read ids, sorted by the order of priviously clustering results
    
    '''
    read_id_index = {i[-1]: n for n, i in enumerate(read_info)}
    mod_results_sorted, read_info_sorted = [], []
    for read_id in read_id_sorted:
            mod_results_sorted.append(mod_results[read_id_index[read_id]])
            read_info_sorted.append(read_info[read_id_index[read_id]])

    return mod_results_sorted, read_info_sorted


def sorted_reads_by_pos(mod_results, read_info, pos: str = 'end'):
    '''Sort read by position

    Parameters
    ----------
    ascending : bool
        if True, sort by ascending order
    '''
    if pos == 'end':
        _read_info = np.array(read_info)[:, 1].astype(int)
        sorted_index = np.argsort(_read_info)[::-1]
    elif pos == 'start':
        _read_info = np.array(read_info)[:, 0].astype(int)
        sorted_index = np.argsort(_read_info)
    else:
        raise ValueError('pos should be either "end" or "start"')

    mod_results, read_info = read_list_sorting([mod_results], [read_info], sorted_index)

    return mod_results[0], read_info[0]


def plot_single_read(ax, mod_results: list, read_info: list, start: int, end: int, read_color: str = '#f5f5f5', methylated_color: str = '#030000', unmethylated_color: Optional[str] = None, xticks_visible: bool = False, read_width: int = 1):
    '''Plot the modification results of a single read
    
    Parameters
    ----------
    ax
        the axis to plot on
    mod_results : list
        a list of numpy arrays, each array is the modification results of a read.
    read_info : list
        a list of lists, each list contains the positions of a read
    start : int
        the start position of the region you want to plot
    end : int
        the end of the region you want to plot
    read_color : str, optional
        the color of the read
    methylated_color : str, optional
        the color of the methylated sites
    unmethylated_color : str, optional
        the color of the unmethylated sites
    xticks_visible : bool, optional
        whether to show the xticks
    read_width: int, default=1
        the width of the read
    '''
    region_length = end - start
    Xaxis = np.arange(region_length)
    y = 0
    for data, pos in zip(mod_results, read_info):
        Yaxis = np.array([y]*region_length)
        ax.plot(pos[:2], [y, y], color=read_color, lw=read_width, zorder=0)

        ax.scatter(
            Xaxis[data > 0],
            Yaxis[data > 0],
            s=read_width,
            marker='s',
            edgecolors='none',
            c=methylated_color,
            zorder=5
        )

        if unmethylated_color is not None:
            ax.scatter(
                Xaxis[data < 0],
                Yaxis[data < 0],
                s=read_width,
                marker='s',
                edgecolors='none',
                c=unmethylated_color,
                zorder=1,
            )
        
        y += -1

    remove_axes(ax)
    ax.set_xlim(0, region_length)
    # ax.set_ylim(y-read_width*2, 0+read_width*2)


def remove_axes(ax):
    ax.spines["right"].set_visible(False)  # 去线
    ax.spines["left"].set_visible(False)  # 去线
    ax.spines["top"].set_visible(False)  # 去线
    ax.spines["bottom"].set_visible(False)
    ax.yaxis.set_major_locator(ticker.NullLocator())  # 去y数字
    ax.xaxis.set_major_locator(ticker.NullLocator())


def plot_genome_coordinate(ax, start, end):
    ax.spines["right"].set_visible(False)  # 去线
    ax.spines["left"].set_visible(False)  # 去线
    ax.spines["top"].set_visible(False)  # 去线
    ax.yaxis.set_major_locator(ticker.NullLocator())  # 去y数字

    ax.set_xlim(0, end-start)

    # plot genome coordinate
    offset = start - start // 1000 * 1000
    xticks = ax.get_xticks()[1 : -1].astype(int) - offset

    ax.set_xticks(xticks)
    ax.set_xticklabels(map(lambda x: format(int(x), ','), xticks+start))


def plot_bw_track(ax, track_data, start, end, track_color: str = None, data_range: tuple = (None, None)):
    '''This function takes in a matplotlib axis, a dataframe of track data, a start and end time, and a
    color, and plots the track on the axis
    
    Parameters
    ----------
    ax
        the axis to plot the track on
    track_data
        a list of tuples, each tuple is a point in the track
    start
        the start time of the track
    end
        the end of the track
    track_color
        the color of the track
    
    '''
    ax.fill_between(np.linspace(start, end, end-start), track_data, color=track_color)
    # set ax
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    if data_range[0] is None:
        ax.spines["left"].set_visible(False)
        ax.yaxis.set_major_locator(ticker.NullLocator())
    else:
        ax.set_yticks([data_range[1]])
        ax.tick_params(axis='both', which='major', labelsize=9)

    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.xaxis.set_ticks_position("none")
    ax.set_xlim(start, end)
    if not(None in data_range):
        ax.set_ylim(data_range)


class Single_Read:
    def __init__(
        self,
        chrom: str,
        start: int,
        end: int,
    ):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.gene_model_n = 0

        # tracks
        self.track_data = []

        # gene model
        self.gene_model = []
        self.gene_properties = {}

        # modified bases results
        self.mod_results = []
        self.read_info = []
        self.mod_tracks_n = 0
        self.methylated_color = []
        self.unmethylated_color = []
        self.read_color = []
        self.mod_track_id = {}
        self.read_width = []

        # bigWig tracks
        self.bw_track_list = []
        self.bw_tracks_n = 0
        self.bw_track_id = {}


    def add_gene_model(
        self, 
        *infiles: tuple, 
        gene_list=None,
        y_space: int = 1,
        track_height: int = 1,
        arrow: bool = True,
        small_arrow: bool = False,
        annotation_pos: bool = 'left',
        gene_color: str = 'k',
    ):
        self.gene_models = get_gene_model(self.chrom, self.start, self.end, *infiles)
        self.gene_model_ylim = get_y_pos_continuous(
            self.gene_models, gene_list=gene_list, threshold=8
        )  # 不重叠y轴的基因数目

        self.gene_properties['arrow'] = arrow
        self.gene_properties['small_arrow'] = small_arrow
        self.gene_properties['annotation_pos'] = annotation_pos

        self.y_space = y_space
        self.track_data.append(('plot_gene_model', self.gene_model_n, track_height))
        self.gene_model_n = 1
        self.gene_color = gene_color


    def add_bw(
        self,
        infile: str,
        color='#5D93C4',
        chrom_prefix='',
        track_height: int = 1,
        data_range: tuple = (None, None),
        label: str = None,
        scale: float = 1,
    ):
        bw = pyBigWig.open(infile)
        chrom = chrom_prefix + self.chrom
        bw_data = bw.values(chrom, self.start, self.end)
        bw_data = np.nan_to_num(bw_data)
        bw_data = bw_data * scale

        self.bw_track_list.append((bw_data, color, data_range))
        self.track_data.append(('plot_bw', self.bw_tracks_n, track_height))
        self.bw_tracks_n += 1

        if label is None:
            label = f'bigWig_{self.bw_tracks_n}'
        self.bw_track_id[label] = self.bw_tracks_n - 1
    

    def add_bw_from_read_data(
        self,
        mod_data,
        read_info,
        color='#5D93C4',
        data_range: tuple = (None, None),
        track_height: int = 1,
        label: str = None,
    ):
        mod_data = np.array(mod_data, dtype=float)
        bw_data = []
        for _mod_data, _read_info in zip(mod_data, read_info):
            read_relative_start = _read_info[0] - self.start if _read_info[1] > self.start else 0
            read_relative_end = _read_info[1] - self.start if _read_info[1] < self.end else self.end - self.start

            _mod_data[: read_relative_start] = np.nan
            _mod_data[read_relative_end: ] = np.nan
            # b[(b < _read_info[0]) | (b > _read_info[1])] = np.nan
            bw_data.append(b)

        bw_data = np.array(bw_data)
        if (bw_data == -1).any():
            bw_data[bw_data == -1] = 0

        bw_data = np.nanmean(bw_data, axis=0)
        self.bw_track_list.append((bw_data, color, data_range))
        self.track_data.append(('plot_bw', self.bw_tracks_n, track_height))
        self.bw_tracks_n += 1

        if label is None:
            label = f'bigWig_{self.bw_tracks_n}'
        self.bw_track_id[label] = self.bw_tracks_n - 1


    def add_mod_results(
        self, 
        infile: str, 
        fully_span: bool=False, 
        left_span: bool=False,
        right_span: bool=False,
        start_before: int=None,
        end_after: int=None,
        sort: bool=False,
        stranded: bool=False,
        expand: int=5, 
        read_set: set = None,
        read_color: str = '#e7e7e7',
        methylated_color: str = '#52af4c',
        unmethylated_color: str = None,
        mod_track_height: int = 4,
        bw_track_height: int = 1,
        filter_read: bool = False,
        add_bigwig: bool = False,
        bw_color: str = '#52af4c',
        data_range: tuple = (None, None),
        read_width: int = 1,
        soft_clip: dict = None
    ):
        '''This function adds a track to the plot that shows the methylation status
        
        Parameters
        ----------
        infile : str
            the path to the file containing the methylation data
        fully_span : bool, optional
            If True, the read will span the entire length of the region
        left_span : bool, optional
            If True, the read will span the entire left end of the region
        right_span : bool, optional
            If True, the read will span the entire right end of the region
        stranded : bool, optional
            If True, the methylation status will be shown on the forward and reverse strand separately
        expand : int, optional
            pseudo int for visulazation
        read_set : set
            a set of reads to be plotted. If None, all reads will be plotted.
        read_color : str, optional
            color of the read
        methylated_color : str, optional
            color of methylated reads
        unmethylated_color : str, optional
            color of unmethylated reads
        filter_read : bool, optional
            if True, only reads that are fully methylated or fully unmethylated will be plotted.
        add_bigwig : bool, optional
            get the bigwig data from the modification files
        bw_color : str, optional
            color of the bigwig track
        data_range : tuple
            tuple = (None, None)
        '''
        
        unmethylated = True if unmethylated_color is not None else False  # only for 5mC data, optional

        if not add_bigwig:
            mod_results, read_info = get_mod_results(
                infile, self.chrom, self.start, self.end, 
                fully_span=fully_span,  left_span=left_span, right_span=right_span,
                start_before=start_before, end_after=end_after,
                expand=expand, unmethylated=unmethylated, filter_read=filter_read, read_set=read_set, soft_clip=soft_clip
            )
        else:
            if stranded:
                mod_results, read_info, bw_data_plus, bw_data_minus = get_mod_results(
                infile, self.chrom, self.start, self.end, 
                fully_span=fully_span,  left_span=left_span, right_span=right_span,
                start_before=start_before, end_after=end_after,
                expand=expand, unmethylated=unmethylated, filter_read=filter_read, read_set=read_set, add_bigwig=True, stranded=True, soft_clip=soft_clip
            )
            else:
                mod_results, read_info, bw_data = get_mod_results(
                infile, self.chrom, self.start, self.end, 
                fully_span=fully_span,  left_span=left_span, right_span=right_span,
                start_before=start_before, end_after=end_after,
                expand=expand, unmethylated=unmethylated, filter_read=filter_read, read_set=read_set, add_bigwig=True, stranded=False, soft_clip=soft_clip
            )
            
        if sort:
            if left_span:
                mod_results, read_info = sorted_reads_by_pos(mod_results, read_info, pos='end')
            elif right_span:
                mod_results, read_info = sorted_reads_by_pos(mod_results, read_info, pos='start')

        if not stranded:
            if add_bigwig:
                self.bw_track_list.append((bw_data, bw_color, data_range))
                self.track_data.append(('plot_bw', self.bw_tracks_n, bw_track_height))
                self.bw_tracks_n += 1

                label = f'bigWig_{self.bw_tracks_n}'
                self.bw_track_id[label] = self.bw_tracks_n - 1

            self.mod_results.append(mod_results)
            self.read_info.append(read_info)
            self.track_data.append(('plot_mod', self.mod_tracks_n, mod_track_height))
            self.mod_tracks_n += 1


            label = f'mod_{self.mod_tracks_n}'
            self.mod_track_id[label] = self.mod_tracks_n - 1

            # color
            self.methylated_color.append(methylated_color)
            self.unmethylated_color.append(unmethylated_color)
            self.read_color.append(read_color)

            # line style
            self.read_width.append(read_width)
        
        else:
            # if sort:
            #     if left_span:
            #         mod_results, read_info = sorted_reads_by_pos(mod_results, read_info, pos='end')
            #     elif right_span:
            #         mod_results, read_info = sorted_reads_by_pos(mod_results, read_info, pos='start')

            mod_results_fwd, read_info_fwd, mod_results_rev, read_info_rev = mod_results_stranded(mod_results, read_info)

            if add_bigwig:
                for bw_data, strand in zip((bw_data_plus, bw_data_minus), ('+', '-')):
                    if strand == '+':
                        self.bw_track_list.append((bw_data, bw_color, data_range))
                    else:
                        self.bw_track_list.append((bw_data, bw_color, data_range[::-1]))

                    self.track_data.append(('plot_bw', self.bw_tracks_n, bw_track_height))
                    self.bw_tracks_n += 1
                    label = f'bigWig_{self.bw_tracks_n}'
                    self.bw_track_id[label] = self.bw_tracks_n - 1
            
            self.mod_results.append(mod_results_fwd)
            self.read_info.append(read_info_fwd)
            self.track_data.append(('plot_mod', self.mod_tracks_n, mod_track_height))
            self.mod_tracks_n += 1
            label = f'mod_{self.mod_tracks_n}'
            self.mod_track_id[label] = self.mod_tracks_n - 1

            self.mod_results.append(mod_results_rev)
            self.read_info.append(read_info_rev)
            self.track_data.append(('plot_mod', self.mod_tracks_n, mod_track_height))
            self.mod_tracks_n += 1
            label = f'mod_{self.mod_tracks_n}'
            self.mod_track_id[label] = self.mod_tracks_n - 1

            for i in range(2):
                # color
                self.methylated_color.append(methylated_color)
                self.unmethylated_color.append(unmethylated_color)
                self.read_color.append(read_color)

                # line style
                self.read_width.append(read_width)
    
            
    def cluster_reads(self):
        combine_mod_results = []
        for item in zip(*self.mod_results):
            combine_mod_results.append(np.concatenate(item))
        self.mod_results, self.read_info = read_list_clustering(combine_mod_results, self.mod_results, self.read_info)

    
    def sorted_reads(self, by: int = 0, ascending: bool = True, ranges: tuple = None):
        '''Sort read by values

        Parameters
        ----------
        by : the index of the mod_result used for sorting
        '''
        if ranges is None:
            mod_results_sum = np.sum(self.mod_results[by], axis=1)
        else:
            mod_results_sum = np.array(self.mod_results[by])
            mod_results_sum = np.sum(mod_results_sum[:, ranges[0]:ranges[1]], axis=1)
        if ascending:
            sorted_index = np.argsort(mod_results_sum)
        else:
            sorted_index = np.argsort(mod_results_sum)[::-1]

        self.mod_results, self.read_info = read_list_sorting(self.mod_results, self.read_info, sorted_index)


    def sorted_reads2(self, ascending: bool = True, ranges: tuple = None):
        '''Sort read by values

        Parameters
        ----------
        by : the index of the mod_result used for sorting
        '''
        for i in range(len(self.mod_results)):
            mod_results = self.mod_results[i]
            read_info = self.read_info[i]
            mod_results_sum = np.sum(mod_results, axis=1)
            if ascending:
                sorted_index = np.argsort(mod_results_sum)
            else:
                sorted_index = np.argsort(mod_results_sum)[::-1]
            
            self.mod_results[i] = [mod_results[i] for i in sorted_index]
            self.read_info[i] = [read_info[i] for i in sorted_index]
            

    def sorted_reads_strand_mod(self, by: int = 0, ascending: bool = True, ranges: tuple = None):
        '''Sort read by values for strand specific mod

        Parameters
        ----------
        by : the index of the mod_result used for sorting
        '''
        if ranges is None:
            mod_results_sum = np.sum(self.mod_results[by], axis=1)
        else:
            mod_results_sum = np.array(self.mod_results[by])
            mod_results_sum = np.sum(mod_results_sum[:, ranges[0]:ranges[1]], axis=1)
        if ascending:
            sorted_index = np.argsort(mod_results_sum)
        else:
            sorted_index = np.argsort(mod_results_sum)[::-1]

        if by % 2 == 0:
            for i in range(0, len(self.mod_results), 2):
                self.mod_results[i] = [self.mod_results[i][j] for j in sorted_index]
                self.read_info[i] = [self.read_info[i][j] for j in sorted_index]
        else:
            for i in range(1, len(self.mod_results), 2):
                self.mod_results[i] = [self.mod_results[i][j] for j in sorted_index]
                self.read_info[i] = [self.read_info[i][j] for j in sorted_index]


    def summary(self):
        print(f'{self.chrom}:{self.start}-{self.end}\nlength: {self.end-self.start}')
        mod_tracks_id = ', '.join(self.mod_track_id.keys())
        print(f'mod_tracks: {mod_tracks_id}')
        bw_tracks_id = ', '.join(self.bw_track_id.keys())
        print(f'bw_tracks: {bw_tracks_id}')
        
    
    def plot(
        self, figsize=(12, 3), 
        hspace: float = .01, plot_order: list = None, xticks: bool=True,
        mod_tracks_set: set = None, bw_tracks_set: set = None,
        subsample: float = 0):
        '''This function plots the tracks

        Parameters
        ----------
        figsize : tuple, optional
            figure size, by default (12, 3)
        hspace : float, optional
            the space between tracks, by default .01
        plot_order : list, optional
            the order of the tracks, by default None
            eg. plot_order=('plot_gene_model', 'plot_bw', 'plot_mod')
        xticks : bool, optional
            if True, xticks will be plotted at the bottom of the figure, by default True
        subsample : float, optional
            if > 0, the reads will be subsampled to the given ratio or count, by default 0
        '''

        if plot_order is None:
            track_data = self.track_data
        else:
            track_data = []
            for order in plot_order:  # plot_order, a list of "plot_type"
                for item in self.track_data:  # self.track_data = [('plot_type', index, track_height), ...]
                    if order == 'plot_mod' and mod_tracks_set is not None:
                        if item[1] not in mod_tracks_set:
                            continue
                    elif order == 'plot_bw' and bw_tracks_set is not None:
                        if item[1] not in bw_tracks_set:
                            continue
                    if item[0] == order:
                        track_data.append(item)
        
        gridspec_kw = [item[2] for item in track_data]  # track height
        gridspec_kw.append(.1)  # xticks tracks

        fig, axes = plt.subplots(
            figsize=figsize, 
            nrows=len(gridspec_kw),
            gridspec_kw={"height_ratios": gridspec_kw},
        )

        for axes_index, item in enumerate(track_data):
            if item[0] == 'plot_gene_model':
                auto_arrow = (self.end - self.start) // 1500
                if auto_arrow == 0:
                    auto_arrow = 1
                plot_gene_model(axes[axes_index], self.gene_models, self.start, self.end, arrow=self.gene_properties['arrow'], small_arrow=self.gene_properties['small_arrow'], annotation_pos=self.gene_properties['annotation_pos'], y_space=self.y_space, auto_arrow=auto_arrow, gene_color=self.gene_color)

            elif item[0] == 'plot_bw':
                if bw_tracks_set is not None and item[1] not in bw_tracks_set:
                    continue

                bw_data, color, data_range = self.bw_track_list[item[1]]  # item[1] is the index of the bw track
                plot_bw_track(axes[axes_index], bw_data, self.start, self.end, color, data_range)
            
            elif item[0] == 'plot_mod':
                if mod_tracks_set is not None and item[1] not in mod_tracks_set:
                    continue

                mod_results = self.mod_results[item[1]]  # item[1] is the index of the mod track
                read_info = self.read_info[item[1]]
                read_color = self.read_color[item[1]]
                methylated_color = self.methylated_color[item[1]]
                unmethylated_color = self.unmethylated_color[item[1]]
                read_width = self.read_width[item[1]]

                xticks_visible = True if axes_index == len(gridspec_kw)-1 else False  # plot tticks or not
                if subsample > 0:
                    mod_results, read_info = subsample_reads(mod_results, read_info, subsample)

                plot_single_read(
                    axes[axes_index], mod_results, read_info, self.start, self.end, 
                    read_color=read_color, methylated_color=methylated_color, unmethylated_color=unmethylated_color,
                    xticks_visible=xticks_visible, read_width=read_width)
        
        if xticks:
            plot_genome_coordinate(axes[axes_index+1], self.start, self.end)
        else:
            remove_axes(axes[axes_index+1])
        

        plt.subplots_adjust(hspace=hspace)

        return axes


    def plot_all(
        self, figsize=(12, 3), 
        gene_track_height=1, bw_track_height=1.5, mod_track_height=7,
        hspace=None, wspace=None,
        ):

        if self.bw_tracks_n != 0 and self.bw_tracks_n != self.mod_tracks_n:
            raise ValueError('bw_tracks_label and mod_tracks_label should have the same size')
        
        gridspec_kw = [gene_track_height] + [bw_track_height] * (self.bw_tracks_n // 2) + [mod_track_height]

        fig, axes = plt.subplots(
            figsize=figsize, 
            nrows=self.gene_model_n + self.bw_tracks_n // 2 + self.mod_tracks_n // 2,
            ncols=self.mod_tracks_n,
            gridspec_kw={"height_ratios": gridspec_kw},
        )
        
        for ax in axes[0]:
            if self.gene_model_n == 1:
                plot_gene_model(ax, self.gene_models, self.start, self.end, arrow=self.arrow, annotation_pos=self.annotation_pos)

        axes_index = 0
        if self.bw_tracks_n > 0:
            axes_index += 1
            for (bw_data, color, data_range), ax in zip(self.bw_track_list, axes[axes_index]):
                plot_bw_track(ax, bw_data, self.start, self.end, color, data_range)

        axes_index += 1
        for (mod_rsults, read_info, methylated_color, unmethylated_color, read_color), ax in zip(
            zip(self.mod_results, self.read_info, self.methylated_color, self.unmethylated_color, self.read_color),
            axes[axes_index]):

            plot_single_read(ax, mod_rsults, read_info, self.start, self.end, read_color, methylated_color, unmethylated_color, xticks_visible=True)

        plt.subplots_adjust(hspace=hspace, wspace=wspace)

        return axes
