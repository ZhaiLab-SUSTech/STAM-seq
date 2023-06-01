from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.patches as mp
import seaborn as sns

# 设置全局字体
font_dirs = ['/public/home/mowp/test/fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'


import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import pysam
from itertools import repeat
import math
import scipy
from scipy import ndimage
from loguru import logger
import pyBigWig
from scipy.interpolate import interp1d


def get_chrom_sizes(chrom_sizes: str, exclude_chroms: set = {}) -> dict:
    '''It reads a file with chromosome sizes and returns a dictionary with chromosome names as keys and
    chromosome sizes as values
    
    Parameters
    ----------
    chrom_sizes : str
        a file containing the chromosome sizes of the genome you're working with.
    exclude_chroms : set
        a set of chromosomes to exclude from the analysis.
    
    Returns
    -------
        A dictionary with the chromosome name as the key and the chromosome size as the value.
    
    '''
    with open(chrom_sizes, 'r') as f:
        chrom_sizes = {}
        for line in f:
            l = line.rstrip().split('\t')
            if l[0] in exclude_chroms:
                continue
            chrom_sizes[l[0]] = int(l[1])

    return chrom_sizes


def get_methyratio_within_bin(
    infile: str, 
    chrom: str, 
    start: int, 
    end: int, 
    canonical: int = 11,  # column of the canonical base within the tabix file
    modification: int = 12,  # column of the modification base within the tabix file
    max_eff_n: int = 4,
    threshold: int = 0,
    return_pos: bool = False,
    return_count: bool = False):

    '''This function takes a tabix file, a chromosome, a start position, and an end position, and returns
    the ratio of methylated bases to total bases within that region
    
    Parameters
    ----------
    infile : str
        the path to the tabix file
    chrom : str
        chromosome
    start : int
        int = start position of the bin
    end : int
        int = end,
    canonical : int, optional
        column of the canonical base within the tabix file
    modification : int, optional
        int = 12,  # column of the modification base within the tabix file
    threshold : int, optional
        minimum number of reads required to call a modification
    
    '''

    tbx = pysam.TabixFile(infile)
    all_n, all_mod, eff_n = 0, 0, 0
    try:
        for line in tbx.fetch(chrom, start, end):
            line = line.rstrip().split('\t')

            n_canon = int(line[canonical])
            n_mod = int(line[modification])

            all_n +=  n_canon + n_mod
            all_mod += n_mod
            if n_canon + n_mod >= threshold:
                eff_n += 1

        if eff_n >= max_eff_n:
            methyratio = all_mod/all_n
        else:
            methyratio = np.nan

    except ValueError:
        raise ValueError(f'Error in {chrom}:{start}-{end}')

    tbx.close()
    if return_count:
        return all_mod, all_n
    if return_pos:
        return methyratio, chrom, start, end
    else:
        return methyratio


def get_methyratio_within_bin_turbo(
    infile: str, 
    chrom_sizes: str, 
    binsize: int=500_000, 
    stepsize: int=500_000,
    threshold: int=0,
    canonical: int = 11,  # column of the canonical base within the tabix file
    modification: int = 12,  # column of the modification base within the tabix file
    exclude_chroms: set = {'ChrM', 'ChrC'}, 
    threads: int=64):

    '''This function takes a tabix file of methylation data, a chromosome size file, and a number of bins,
    and returns a dataframe of the methylation ratio within each bin
    
    Parameters
    ----------
    infile : str
        str = path to the tabix file
    chrom_size : str
        a file with the chromosome sizes
    binsize : int, optional
        the size of the bins to use for the analysis
    threshold : int, optional
        minimum number of reads required to call a base as modified
    canonical : int, optional
        column of the canonical base within the tabix file
    modification : int, optional
        int = 12,  # column of the modification base within the tabix file
    exclude_chroms : set
        set = {'chrM', 'chrC'},
    threads : int, optional
        number of threads to use
    
    '''

    chrom_sizes = get_chrom_sizes(chrom_sizes, exclude_chroms=exclude_chroms)

    methyratio = {}
    for chrom, chrom_size in chrom_sizes.items():
        start, end = [], []
        for i in range(0, chrom_size, stepsize):
            if i + binsize <= chrom_size:
                start.append(i)
                end.append(i + binsize)
            else:
                start.append(chrom_size - binsize)
                end.append(chrom_size)
                break

        with ProcessPoolExecutor(max_workers=threads) as e:
            chunksize = math.ceil(len(start) / threads)
            results = e.map(
                get_methyratio_within_bin,
                repeat(infile),
                repeat(chrom),
                start,
                end,
                repeat(canonical),
                repeat(modification),
                repeat(threshold),
                chunksize=chunksize)
        
        methyratio[chrom] = list(results)
    
    return methyratio


def get_values_within_bin(
    infile: str, 
    chrom: str, 
    start: int, 
    end: int, 
    pos: int = 11,  # column of the values within the tabix file
    ):

    tbx = pysam.TabixFile(infile)
    values = []
    for line in tbx.fetch(chrom, start, end):
        line = line.rstrip().split('\t')
        values.append(int(line[pos]))

    tbx.close()
    return np.sum(values) / (end - start)


def get_values_within_bin_turbo(
    infile: str, 
    chrom_sizes: str, 
    binsize: int=500_000, 
    stepsize: int=500_000,
    pos: int = 9,
    exclude_chroms: set = {'ChrM', 'ChrC'}, 
    threads: int=64):

    chrom_sizes = get_chrom_sizes(chrom_sizes, exclude_chroms=exclude_chroms)

    values = {}
    for chrom, chrom_size in chrom_sizes.items():
        start, end = [], []
        for i in range(0, chrom_size, stepsize):
            if i + binsize <= chrom_size:
                start.append(i)
                end.append(i + binsize)
            else:
                start.append(chrom_size - binsize)
                end.append(chrom_size)
                break

        with ProcessPoolExecutor(max_workers=threads) as e:
            chunksize = math.ceil(len(start) / threads)
            results = e.map(
                get_values_within_bin,
                repeat(infile),
                repeat(chrom),
                start,
                end,
                repeat(pos),
                chunksize=chunksize)
        
        values[chrom] = list(results)
    
    return values
    

def plot_metaplot_genome(
    methyratio,
    chrom_size: str,
    ax=None,
    figsize:tuple = (10, 2),
    bins:int = 10_000, 
    ylabel:str = 'methylation ratio',
    color:str = None,
    lw: int = 1,
    size_factor: float = None,
    label: str = None,
    ls: str = None,
    exclude_chroms: set = {'ChrM', 'ChrC'}, 
):

    with open(chrom_size, 'r') as f:
        chrom_size = {}
        for line in f:
            l = line.rstrip().split('\t')
            if l[0] in exclude_chroms:
                continue
            chrom_size[l[0]] = int(l[1])

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=figsize, sharey=True)

    for i, chrom in enumerate(methyratio):
        ax_chr = ax[i]
        data = np.array(methyratio[chrom])

        if size_factor is not None:
            data = data / size_factor
            

        ax_chr.plot(data, color=color, linewidth=lw, label=label, ls=ls)
        ax_chr.set_xticks([0, chrom_size[chrom]//bins])
        ax_chr.set_xticklabels([0, chrom_size[chrom] // 1_000_000])
        ax_chr.title.set_text(chrom)
        sns.despine(ax=ax_chr)

    ax[0].set_ylabel(ylabel)
    
    return ax


def plot_distribution_region(
    methyratio: dict,
    regions: dict,
    ax=None,
    figsize:tuple = (10, 2),
    binsize:int = 10_000, 
    extend: int = 2_000_000,
    ylabel:str = 'methylation ratio',
    color:str = None,
    lw: int = 1,
    size_factor: float = None,
    label: str = None,
    ls: str = None,
):

    ncols = len(methyratio)
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)

    for i, chrom in enumerate(methyratio):
        ax_chr = ax[i]
        data = np.array(methyratio[chrom])

        if size_factor is not None:
            data = data / size_factor
            
        ax_chr.plot(data, color=color, linewidth=lw, label=label, ls=ls)
        region = np.array(regions[chrom]) / binsize
        region[0] += -extend / binsize
        region[1] += extend / binsize
        ax_chr.set_xlim(region)
        xticks = ax_chr.get_xticks()
        ax_chr.set_xticks(xticks)
        ax_chr.set_xticklabels((xticks * binsize / 1_000_000).astype(int))
        ax_chr.set_xlim(region)
        ax_chr.title.set_text(chrom)
        sns.despine(ax=ax_chr)

    ax[0].set_ylabel(ylabel)
    
    return ax


def plot_cen_range(
    ax, bins:int, cen_region: tuple, 
    height_ratio: float = .02, color: str = '#555555'):
    ylim = ax.get_ylim()
    height = height_ratio*(ylim[1]-ylim[0])
    cen_region = np.array(cen_region)
    ax.fill_between(cen_region//bins, (ylim[0]+height, ylim[0]+height), (ylim[0], ylim[0]), color=color)


def tabix_stats(
    infile, chrom, start: int = None, end: int = None, 
    canonical: int = 11,  modification: int = 12):
    '''
    Get the average methylation ratio within a region
    
    Parameters
    ----------
    infile
        the name of the file you want to get stats from
    chrom
        chromosome
    start : int
        int = None, end: int = None
    end : int
        int = None,
    canonical : int, optional
        the column number of the canonical transcript
    modification : int, optional
        the column in the file that contains the modification type
    
    '''

    tbx = pysam.TabixFile(infile)
    all_n, all_mod = 0, 0
    for line in tbx.fetch(chrom, start, end):
        line = line.rstrip().split('\t')

        n_canon = int(line[canonical])
        n_mod = int(line[modification])

        all_n +=  n_canon + n_mod
        all_mod += n_mod

    methyratio = all_mod / all_n
    tbx.close()

    return methyratio


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def tabix_value(infile, chrom, start, end, binsize, canonical, modification, return_count: bool = False):
    _start = np.arange(start, end, binsize)
    _end = _start + binsize
    cov = map(
        get_methyratio_within_bin, 
        repeat(infile), 
        repeat(chrom), _start, _end,
        repeat(canonical), repeat(modification), repeat(4), repeat(0), repeat(False), repeat(return_count),
    )

    if return_count:
        cov = np.array(list(cov))
        all_mod = cov[:, 0]
        all_n = cov[:, 1]
        return all_mod, all_n

    cov = list(cov)
    return cov


def tabix_scale(
    infile: str, 
    chrom: str, site1: int, site2: int,  strand: str,
    chrom_sizes: dict,
    before: int = 1000, after: int = 1000, regionbody: int = 1000,
    binsize: int = 100,
    canonical: int = 11,  modification: int = 12,
    return_count: bool = False):

    if strand == '+':
        start = site1 - before
        end = site2 + after
    elif strand == '-':
        start = site1 - after
        end = site2 + before
    else:
        raise ValueError('strand must be + or -')
    
    if start < 0 and end > chrom_sizes[chrom]:
        return

    if return_count:
        if strand == '+':
            # 5' coverage
            cov_5_mod, cov_5_n = tabix_value(infile, chrom, start, site1, binsize, canonical, modification, return_count)
            # 3' coverage
            cov_3_mod, cov_3_n = tabix_value(infile, chrom, site2, end, binsize, canonical, modification, return_count)
            # genebody coverage
            cov_gb_mod, cov_gb_n = tabix_value(infile, chrom, site1, site2, binsize, canonical, modification, return_count)
        
        else:
            # 5' coverage
            cov_5_mod, cov_5_n = tabix_value(infile, chrom, site2, end, binsize, canonical, modification, return_count)
            cov_5_mod = cov_5_mod[::-1]
            cov_5_n = cov_5_n[::-1]
            # 3' coverage
            cov_3_mod, cov_3_n = tabix_value(infile, chrom, start, site1, binsize, canonical, modification, return_count)
            cov_3_mod = cov_3_mod[::-1]
            cov_3_n = cov_3_n[::-1]
            # genebody coverage
            cov_gb_mod, cov_gb_n = tabix_value(infile, chrom, site1, site2, binsize, canonical, modification, return_count)
            cov_gb_mod = cov_gb_mod[::-1]
            cov_gb_n = cov_gb_n[::-1]

        zoom = regionbody / binsize / len(cov_gb_mod),
        cov_gb_mod = scipy.ndimage.zoom(cov_gb_mod, zoom, order=0, mode='nearest')
        cov_gb_n = scipy.ndimage.zoom(cov_gb_n, zoom, order=0, mode='nearest')

        cov_mod = np.concatenate([cov_5_mod, cov_gb_mod, cov_3_mod])
        cov_n = np.concatenate([cov_5_n, cov_gb_n, cov_3_n])

        return cov_mod, cov_n


    if strand == '+':
        # 5' coverage
        cov_5 = tabix_value(infile, chrom, start, site1, binsize, canonical, modification)
        # 3' coverage
        cov_3 = tabix_value(infile, chrom, site2, end, binsize, canonical, modification)
        # genebody coverage
        cov_gb = tabix_value(infile, chrom, site1, site2, binsize, canonical, modification)
    
    else:
        # 5' coverage
        cov_5 = tabix_value(infile, chrom, site2, end, binsize, canonical, modification)[::-1]
        # 3' coverage
        cov_3 = tabix_value(infile, chrom, start, site1, binsize, canonical, modification)[::-1]
        # genebody coverage
        cov_gb = tabix_value(infile, chrom, site1, site2, binsize, canonical, modification)[::-1]

    cov_gb = scipy.ndimage.zoom(
        cov_gb,
        regionbody / binsize / len(cov_gb),
        order=0,
        mode='nearest')
    
    cov = np.concatenate([cov_5, cov_gb, cov_3])

    return cov


def tabix_scale_region(
    infile: str,
    site_info: list,
    chrom_sizes: str,
    before: int = 1000, after : int = 1000, regionbody : int = 1000, 
    binsize: int = 100,
    canonical: int = 11,  modification: int = 12, return_count: bool = False,
    threads = 64):

    site_info = np.array(site_info)
    chrom = site_info[:, 0]
    site1 = site_info[:, 1].astype('int')
    site2 = site_info[:, 2].astype('int')
    strand = site_info[:, 3]

    chrom_sizes = get_chrom_sizes(chrom_sizes)

    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = math.ceil(len(site_info) / threads)
        results = e.map(
            tabix_scale,
            repeat(infile),
            chrom, site1, site2, strand,
            repeat(chrom_sizes),
            repeat(before), repeat(after), repeat(regionbody), repeat(binsize),
            repeat(canonical), repeat(modification), repeat(return_count),
            chunksize=chunksize)

    if return_count:
        cov_mod, cov_n = [], []
        for _cov_mod, _cov_n in results:
            cov_mod.append(_cov_mod)
            cov_n.append(_cov_n)
        
        cov_mod = np.nanmean(np.array(cov_mod), axis=0)
        cov_n = np.nanmean(np.array(cov_n), axis=0)

        return cov_mod / cov_n
    
    cov = []
    n = 0
    for _cov in results:
        if _cov is not None:
            cov.append(_cov)
            n += 1
    
    cov = np.nanmean(cov, axis=0)
    logger.info(f'n = {n}')

    return cov


def tabix_site_cov(
    infile: str,
    chrom: str, site: int, strand: str,
    before: int = 1000, after : int = 1000, binsize: int = 100,
    canonical: int = 11,  modification: int = 12):

    site = int(site)
    chrom = str(chrom)
    if strand == '+':
        start = site - before
        end = site + after
    else:
        start = site - after
        end = site + before
    
    cov_mod, cov_n = tabix_value(infile, chrom, start, end, binsize, canonical, modification, return_count=True)
    if strand == '-':
        cov_mod = cov_mod[::-1]
        cov_n = cov_n[::-1]
    
    return cov_mod, cov_n



def tabix_reference_point(
    infile: str,
    site_info: list,
    before: int = 1000, after : int = 1000, binsize: int = 100,
    canonical: int = 11,  modification: int = 12,
    threads = 64):

    chrom = site_info[:, 0]
    site = site_info[:, 1].astype('int')
    strand = site_info[:, 2]
    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = math.ceil(len(site_info) / threads)
        results = e.map(
            tabix_site_cov,
            repeat(infile),
            chrom, site, strand,
            repeat(before), repeat(after), repeat(binsize),
            repeat(canonical), repeat(modification),
            chunksize=chunksize)
    
    cov_mod, cov_n = [], []
    for _cov_mod, _cov_n in results:
        cov_mod.append(_cov_mod)
        cov_n.append(_cov_n)
    
    cov_mod = np.nanmean(np.array(cov_mod), axis=0)
    cov_n = np.nanmean(np.array(cov_n), axis=0)

    x = cov_mod / cov_n
    not_nan = np.logical_not(np.isnan(x))
    indices = np.arange(len(x))
    interp = interp1d(indices[not_nan], x[not_nan])
    x = interp(indices) 

    return x


def bigwig_within_bin(infile: str, chrom: str, chromsize: str, binsize: int, stepsize: int, maxvalue: int):

    bwfile = pyBigWig.open(infile)
    methyratio = []
    is_break = False
    for i in range(0, chromsize, stepsize):
        i2 = i+binsize if i+binsize <= chromsize else chromsize
        if i2 >= chromsize:
            i = chromsize - binsize
            i2 = chromsize - 1
            is_break = True
        
        try:
            values = bwfile.stats(chrom, i, i2)[0]
        except RuntimeError:
            raise(RuntimeError(f'{chrom} {i} {i2}'))
        
        if values > maxvalue:
            values = maxvalue

        methyratio.append(values)

        if is_break:
            break
    
    bwfile.close()
    
    return methyratio, chrom


def genome_wide_bigwig_within_bin(infile: str, chromsize, binsize: int = 100_000, stepsize: int = 10_000, maxvalue: int = 100, exclude_chr: set = {'ChrM', 'ChrC'}):
    chroms, chromsizes = [], []
    with open(chromsize, 'r') as f:
        for line in f:
            line = line.rstrip().split('\t')
            if line[0] in exclude_chr:
                continue

            chroms.append(line[0])
            chromsizes.append(int(line[1]))
    
    with ProcessPoolExecutor(max_workers=len(chroms)) as e:
        results = e.map(
            bigwig_within_bin,
            repeat(infile),
            chroms, chromsizes, 
            repeat(binsize), repeat(stepsize), repeat(maxvalue))
    
    coverage = {}
    for cov, chrom in results:
        coverage[chrom] = cov
    
    return coverage


def bigwig_range(infile: str, chrom: str, start: int, end: int, binsize: int = 10, maxvalue: int = None):
    '''
    get methylation ratio within a range

    Parameters
    ----------
    infile : str
        bigwig file
    chrom : str
        chromosome
    start : int
        start position
    end : int
        end position
    binsize : int, optional
        binsize, by default 10
    
    Returns
    -------
    list
        methylation ratio
    '''
    values = []
    bwfile = pyBigWig.open(infile)
    for i in range(start, end, binsize):
        i2 = i+binsize if i+binsize <= end else end
        try:
            _vaules = bwfile.stats(chrom, i, i2)[0]
            if _vaules is None:
                _vaules = 0
            if maxvalue is not None and values > maxvalue:
                values = maxvalue
            values.append(_vaules)
        except RuntimeError:
            raise(RuntimeError(f'{chrom} {i} {i2}'))
    bwfile.close()

    return np.array(values)


def range_bigwig_within_bin(infile: str, ranges: list, binsize: int = 100_000, stepsize: int = 10_000, maxvalue: int = None):
    chroms = ranges[:, 0]
    starts = ranges[:, 1].astype('int')
    ends = ranges[:, 2].astype('int')
    
    with ProcessPoolExecutor(max_workers=len(chroms)) as e:
        results = e.map(
            bigwig_range,
            repeat(infile),
            chroms, starts, ends,
            repeat(binsize), repeat(maxvalue))
    
    coverage = {}
    for cov, chrom in zip(results, chroms):
        coverage[chrom] = cov
    
    return coverage
        


def single_read_mod_ratio(infile: str, chrom: str, start: int, end: int):
    results = []
    n = 0
    tbx = pysam.TabixFile(infile)
    for line in tbx.fetch(chrom, start, end):
        read_id, mod, _chrom, _start, _end, strand, methylated_pos, unmethylated_pos = line.split('\t')

        _start = int(_start)
        _end = int(_end)
        methylated_pos = np.fromstring(methylated_pos, sep=',', dtype=int)
        unmethylated_pos = np.fromstring(unmethylated_pos, sep=',', dtype=int)

        methylated_pos = methylated_pos[((methylated_pos >= start) & (methylated_pos <= end))]
        unmethylated_pos = unmethylated_pos[((unmethylated_pos >= start) & (unmethylated_pos <= end))]

        methylated_n = len(methylated_pos)
        unmethylated_n = len(unmethylated_pos)
        results.append((read_id, methylated_n, unmethylated_n))
        n += 1
    
    # print(n)
    tbx.close()

    results = pd.DataFrame(results, columns=['read_id', 'methylated_n', 'unmethylated_n'])
    return results


def subtel_methy_tl(
    infile: str, 
    chrom: str, start: int, end: int, sub_start: int, sub_end: int,
    left_span: bool = False, right_span: bool = False, 
    start_before: int = None, end_after: int = None,  read_set: set = None):
    """
    Get the telomere length and subtelomere methylation.

    Parameters
    ----------
    infile : str
        Path to the input tabix file.
    chrom : str
        Chromosome name.
    start : int
        Start position of the telomere.
    end : int
        End position of the telomere.
    sub_start : int
        Start position of the subtelomere.
    sub_end : int
        End position of the subtelomere.
    left_span : bool, optional
        Whether to span the left telomere, by default False
    right_span : bool, optional
        Whether to span the right telomere, by default False
    start_before : int, optional
        Start position of the telomere before the span, by default None
    end_after : int, optional
        End position of the telomere after the span, by default None
    read_set : set, optional
        Set of read names to filter, by default None
    """

    results = []
    tbx = pysam.TabixFile(infile)
    for line in tbx.fetch(chrom, start, end):
        read_id, mod, _chrom, _start, _end, strand, methylated_pos, unmethylated_pos = line.split('\t')
        # filter by read_id
        if read_set is not None and read_id not in read_set:
            continue

        _start = int(_start)
        _end = int(_end)
        methylated_pos = np.fromstring(methylated_pos, sep=',', dtype=int)
        unmethylated_pos = np.fromstring(unmethylated_pos, sep=',', dtype=int)

        if left_span and not(_start <= start):
            continue
        if right_span and not(_end > end):
            continue
        if start_before is not None and not(_start <= start_before):
            continue
        if end_after is not None and not(_end > end_after):
            continue
        if not(_start <= sub_start and _end >= sub_end):
            continue

        # if read_id == '336e75f1-3dfb-474e-92e9-6fd4c1f9f308':
        #     print(line)

        methylated_n = len(methylated_pos[((methylated_pos >= sub_start) & (methylated_pos <= sub_end))])
        unmethylated_n = len(unmethylated_pos[((unmethylated_pos >= sub_start) & (unmethylated_pos <= sub_end))])
        all_n = methylated_n + unmethylated_n

        if right_span and start_before is not None:
            tel_len = start_before -_start
        elif left_span and end_after is not None:
            tel_len = _end - end_after
        else:
            raise ValueError('right_span and start_before or left_span and end_after must be set')
        
        if all_n > 0:
            results.append((read_id, methylated_n/all_n, tel_len, methylated_n, all_n, strand))

    return results
