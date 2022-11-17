#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pyBigWig
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pysam
import scipy
from scipy import ndimage
import math
from utils import get_methyratio_within_bin


def get_bin_cov(data: list, bins: int):
    data = np.array(data)
    # data = np.nan_to_num(data)
    
    if bins == 1:
        return data
    if bins < 1:
        raise ValueError('bins must be greater than 1')

    results = []
    for i in range(0, len(data), bins):
        bin_data = data[i:i + bins]
        if np.isnan(bin_data).all():
            results.append(np.nan)
        else:
            mean_bin_data = np.nanmean(bin_data)
            results.append(mean_bin_data)

    return results


################
# For bw file  #
################

# site-point
def bw_site_cov(
    infile: str, 
    chrom: str, site: int, strand: str,
    before: int = 1000, after : int = 1000,
    bins: int = 100,
    chrom_prefix: str = '',
    normalized: str = 'density',
    exclude_chr = None,
    min_coverage: float = 0.0,
    ):
    '''
    Args:
        infile: path to bigWig file
        chrom: chromosome name
        site: target site
        strand: '+' or '-'
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        regionbody: distance in bases to which all regions will be fit
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of given regions
    '''
    site = int(site)
    chrom = str(chrom)

    if exclude_chr is not None and chrom in exclude_chr:
        return
    chrom = chrom_prefix + chrom
        
    bwfile = pyBigWig.open(infile)
    if strand == '+':
        start = site - before
        end = site + after
    else:
        start = site - after
        end = site + before
    if start < 0 or end > bwfile.chroms()[chrom]:
        # remove Invalid interval
        return

    if strand == '+':
        values = bwfile.values(
            chrom, 
            start,
            end)
        cov = get_bin_cov(values, bins)

    elif strand == '-':
        values = bwfile.values(
            chrom, 
            start,
            end)[::-1]
        cov = get_bin_cov(values, bins)
    else:
        return ValueError('strand must be "+" or "-"')
    
    # cov = np.nan_to_num(cov)
    if np.nansum(cov) > min_coverage:
        if normalized == 'density':
            cov = cov / np.nansum(cov)  # density
        elif normalized == 'max':
            cov = cov / np.nanmax(cov)
        return cov

    
def bw_reference_point(
    infile: str, 
    site_info: list,
    before: int = 1000, after : int = 1000,
    bins: int = 100,
    chrom_prefix: str = '',
    normalized: str = 'density',
    exclude_chr = None,
    min_coverage: float = 0.0,
    threads=64):
    '''
    Reference-point refers to a position within a BED region (e.g., the starting point). In this mode, only those genomicpositions before (upstream) and/or after (downstream) of the reference point will be used.

    Args:
        infile: path to bigWig file
        site_info: [(chrom, site, strand), ...]
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    chrom = site_info[:, 0]
    site = site_info[:, 1]
    strand = site_info[:, 2]
    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = int(len(site_info) / threads)
        results = e.map(
            bw_site_cov,
            repeat(infile),
            chrom,
            site,
            strand,
            repeat(before),
            repeat(after),
            repeat(bins),
            repeat(chrom_prefix),
            repeat(normalized),
            repeat(exclude_chr),
            repeat(min_coverage),
            chunksize=chunksize)

    cov = []
    n = 0
    for cov_ in results:
        if cov_ is not None:
            cov.append(cov_)
            n += 1

    cov = np.nanmean(cov, axis=0)
    print(f'n = {n}')
    return cov


# scale region
def bw_scale_cov(
    infile: str, 
    chrom: str, site1: int, site2: int, strand: str,
    before: int = 1000, after : int = 1000, regionbody : int = 1000, 
    bins: int = 100,
    split: bool = False,
    chrom_prefix: str = '',
    normalized: str = 'density',
    exclude_chr = None,
    min_coverage: float = 0.0,
    ):
    '''
    Args:
        infile: path to bigWig file
        chrom: chromosome name
        site1: 5' site
        site2: 3' site
        strand: '+' or '-'
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        regionbody: distance in bases to which all regions will be fit
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        normalized: normalization method, 'density' or 'count'
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    site1 = int(site1)
    site2 = int(site2)
    chrom = str(chrom)

    if exclude_chr is not None and chrom in exclude_chr:
        return

    chrom = chrom_prefix + chrom

    bwfile = pyBigWig.open(infile)
    if strand == '+':
        start = site1 - before
        end = site2 + after
    else:
        start = site1 - after
        end = site2 + before
    if start < 0 or end > bwfile.chroms()[chrom]:
        # remove Invalid interval
        return

    if split:
        # in this mode, regionbody is ignored
        if strand == '+':
            cov_5 = bwfile.values(chrom, site1 - before, site1 + after)
            cov_5 = get_bin_cov(cov_5, bins)
            cov_3 = bwfile.values(chrom, site2 - before, site2 + after)
            cov_3 = get_bin_cov(cov_3, bins)

        elif strand == '-':
            cov_5 = bwfile.values(chrom, site2 - after, site2 + before)[::-1]
            cov_5 = get_bin_cov(cov_5, bins)
            cov_3 = bwfile.values(chrom, site1 - after, site1 + before)[::-1]
            cov_3 = get_bin_cov(cov_3, bins)
        
        cov_5 = np.nan_to_num(cov_5)
        cov_3 = np.nan_to_num(cov_3)

        sum_cov = sum(cov_5)+sum(cov_3)
        if sum_cov > 0:
            # density
            cov_5 = cov_5 / sum_cov
            cov_3 = cov_3 / sum_cov
            return cov_5, cov_3

    else:
        if strand == '+':
            start = site1 - before
            end = site2 + after
        else:
            start = site1 - after
            end = site2 + before
        
        if start < 0 or end > bwfile.chroms()[chrom]:
            return

        if strand == '+':
            cov_5 = bwfile.values(chrom, start, site1)
            cov_5 = get_bin_cov(cov_5, bins)
            cov_3 = bwfile.values(chrom, site2, end)
            cov_3 = get_bin_cov(cov_3, bins)
            # gene_body_region
            cov_gb = bwfile.values(chrom, site1, site2)
            cov_gb = scipy.ndimage.zoom(
                cov_gb,
                regionbody / len(cov_gb),
                order=0,
                mode='nearest')
            cov_gb = get_bin_cov(cov_gb, bins)

        elif strand == '-':
            cov_5 = bwfile.values(chrom, site2, end)[::-1]
            cov_5 = get_bin_cov(cov_5, bins)
            cov_3 = bwfile.values(chrom, start, site1)[::-1]
            cov_3 = get_bin_cov(cov_3, bins)
            # gene_body_region
            cov_gb = bwfile.values(chrom, site1, site2)[::-1]
            cov_gb = scipy.ndimage.zoom(
                cov_gb,
                regionbody / len(cov_gb),
                order=0,
                mode='nearest')
            cov_gb = get_bin_cov(cov_gb, bins)
        else:
            raise ValueError('strand must be "-" or "+"')

        cov = np.concatenate([cov_5, cov_gb, cov_3])
        # cov = np.nan_to_num(cov)

        if np.nansum(cov) > min_coverage:
            if normalized == 'density':
                cov = cov / np.nansum(cov)  # density
            return cov


def bw_scale_regions(
    infile: str, 
    site_info: list,
    before: int = 1000, after : int = 1000, regionbody : int = 1000, 
    bins: int = 100,
    split: bool = False,
    chrom_prefix: str = '',
    normalized: str = 'density',
    exclude_chr = None,
    min_coverage: float = 0.0,
    return_raw: bool = False,
    threads=64):
    '''
    In the scale-regions mode, all regions in the BED file are stretched or shrunken to the length (in bases) indicated by the user.

    Args:
        infile: path to bigWig file
        site_info: [(chrom, site1, site2, strand), ...]
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        regionbody: distance in bases to which all regions will be fit
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        normalized: normalization method, 'density' or 'count'
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    site_info = np.array(site_info)
    chrom = site_info[:, 0]
    site1 = site_info[:, 1]
    site2 = site_info[:, 2]
    strand = site_info[:, 3]

    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = math.ceil(len(site_info) / threads)
        results = e.map(
            bw_scale_cov,
            repeat(infile),
            chrom,
            site1,
            site2,
            strand,
            repeat(before),
            repeat(after),
            repeat(regionbody),
            repeat(bins),
            repeat(split),
            repeat(chrom_prefix),
            repeat(normalized),
            repeat(exclude_chr),
            repeat(min_coverage),
            chunksize=chunksize)

    if split:
        cov_5, cov_3 = [], []
        for res in results:
            if res is not None:
                cov_5_, cov_3_ = res
                cov_5.append(cov_5_)
                cov_3.append(cov_3_)

        cov_5 = np.nanmean(cov_5, axis=0)
        cov_3 = np.nanmean(cov_3, axis=0)

        return cov_5, cov_3

    else:
        cov = []
        n = 0
        for cov_ in results:
            if cov_ is not None:
                cov.append(cov_)
                n += 1

        print(f'n = {n}')
        if return_raw:
            return cov
            
        cov = np.nanmean(cov, axis=0)
        return cov


################
# For bam file #
################

# site-point
def bam_site_cov(
    infile: str, 
    chrom: str, site: int, strand: str, gene_id: str,
    before: int = 1000, after : int = 1000,
    bins: int = 100,
    min_counts: int = 1,
    chrom_prefix: str = '',
    exclude_chr: set = None,
    return_raw: bool = False,
):
    """
    BAM file for tagged FLEP-seq data
    Ignore splicing junction

    Args:
        infile: path to bigWig file
        chrom: chromosome name
        site: target site
        strand: '+' or '-'
        gene_id: gene id
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        min_counts: minimum number of the reads within given region
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of given region
    """
    site = int(site)
    chrom = str(chrom)
    strand_is_reverse = False if strand == '+' else True

    if exclude_chr is not None and chrom in exclude_chr:
        return
    chrom = chrom_prefix + chrom
    
    n = 0

    cov = np.zeros(before+after)

    if strand == '+':
        start = site-before
        end = site+after
    else:
        start = site-after
        end = site+before
    

    with pysam.AlignmentFile(infile, 'rb') as inbam:
        if start < 0 or end > inbam.get_reference_length(chrom):
            return
            
        for read in inbam.fetch(chrom, start, end):
            # 判断是否跟基因是同个方向，针对于链特异文库
            if read.is_reverse != strand_is_reverse:
                continue
            
            read_gene_id = read.get_tag('gi')
            if read_gene_id not in {gene_id, 'None'}:
                continue
                
            if strand == '+':
                read_five_end = read.reference_start
                read_three_end = read.reference_end
                cov_start = read_five_end-start if read_five_end-start >= 0 else 0
                cov_end = read_three_end-start if read_three_end-start <= before+after else end-start
            else:
                read_five_end = read.reference_end
                read_three_end = read.reference_start
                cov_start = end-read_five_end if end-read_five_end >= 0 else 0
                cov_end = end-read_three_end if end-read_three_end <= before+after else end-start

            cov[cov_start: cov_end] += 1            
            n += 1
    
    if return_raw:
        return cov

    if n > min_counts:
        if bins > 1:
            cov = get_bin_cov(cov, bins)
        cov = cov / sum(cov)
        return cov


def bam_reference_point(
    infile: str, 
    site_info: list,
    before: int = 1000, after : int = 1000,
    bins: int = 100,
    min_counts: int = 1,
    chrom_prefix: str = '',
    exclude_chr = None,
    threads=64):
    '''
    Reference-point refers to a position within a BED region (e.g., the starting point). In this mode, only those genomicpositions before (upstream) and/or after (downstream) of the reference point will be used.

    Args:
        infile: path to bigWig file
        site_info: [(chrom, site, strand, gene_id), ...]
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        min_counts: minimum number of the reads
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    chrom = site_info[:, 0]
    site = site_info[:, 1]
    strand = site_info[:, 2]
    gene_id = site_info[:, 3]

    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = int(len(site_info)/threads)
        results = e.map(
            bam_site_cov, 
            repeat(infile),
            chrom,
            site,
            strand,
            gene_id,
            repeat(before),
            repeat(after),
            repeat(bins),
            repeat(min_counts),
            repeat(chrom_prefix),
            repeat(exclude_chr),
            chunksize=chunksize)
    
    cov = []
    n = 0
    for res in results:
        if res is not None:
            cov.append(res)
    
    cov = np.nanmean(cov, axis=0)
    return cov


# scale region
def bam_scale_cov(
    infile: str, 
    chrom: str, site1: int, site2: int, strand: str, gene_id: str,
    before: int = 1000, after : int = 1000, regionbody : int = 1000, 
    bins: int = 100,
    split: bool = False,
    min_counts: int = 1,
    chrom_prefix: str = '',
    exclude_chr = None
    ):
    '''
    Args:
        infile: path to BAM file
        chrom: chromosome name
        site1: 5' site
        site2: 3' site
        strand: '+' or '-'
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        regionbody: distance in bases to which all regions will be fit
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        split: split mode
        min_count: minimum number of reads in the region
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    site1 = int(site1)
    site2 = int(site2)
    region_len = site2-site1
    chrom = str(chrom)
    strand_is_reverse = False if strand == '+' else True

    if exclude_chr is not None and chrom in exclude_chr:
        return

    chrom = chrom_prefix + chrom

    if split:
        cov1 = bam_site_cov(infile, chrom, site1, strand, gene_id, before=before, after=after, bins=bins, return_raw=True)
        cov2 = bam_site_cov(infile, chrom, site2, strand, gene_id, before=before, after=after, bins=bins, return_raw=True)

        if cov1 is None or cov2 is None:
            return
            
        sum_cov = sum(cov1) + sum(cov2)
        if sum_cov > 0:
            cov1 = cov1 / sum_cov
            cov2 = cov2 / sum_cov
            if strand == '+':
                return cov1, cov2
            elif strand == '-':
                return cov2, cov1
            
    else:
        if strand == '+':
            start = site1-before
            end = site2+after
        else:
            start = site1-after
            end = site2+before

        cov = np.zeros(before+region_len+after)

        with pysam.AlignmentFile(infile, 'rb') as inbam:
            if start < 0 or end > inbam.get_reference_length(chrom):
                return
            n = 0
            for read in inbam.fetch(chrom, start, end):
                # 判断是否跟基因是同个方向，针对于链特异文库
                if read.is_reverse != strand_is_reverse:
                    continue
                    
                read_gene_id = read.get_tag('gi')
                if read_gene_id not in {gene_id, 'None'}:
                    continue
                    
                if strand == '+':
                    read_five_end = read.reference_start
                    read_three_end = read.reference_end
                    cov_start = read_five_end-start if read_five_end-start >= 0 else 0
                    cov_end = read_three_end-start if read_three_end-start <= before+after+region_len else end-start
                else:
                    read_five_end = read.reference_end
                    read_three_end = read.reference_start
                    cov_start = end-read_five_end if end-read_five_end >= 0 else 0
                    cov_end = end-read_three_end if end-read_three_end <= before+after+region_len else end-start

                cov[cov_start: cov_end] += 1            
                n += 1
            
            if n > min_counts:
                cov_gb = cov[before: before+region_len]
                cov_gb = scipy.ndimage.zoom(
                    cov_gb,
                    regionbody / len(cov_gb),
                    order=0,
                    mode='nearest')
                cov_5 = cov[ : before]
                cov_3 = cov[before+region_len : ]
                cov = np.concatenate([cov_5, cov_gb, cov_3])

                if bins > 1:
                    cov = get_bin_cov(cov, bins)
                cov = cov / sum(cov)
                
                return cov


def bam_scale_region(
    infile: str, 
    site_info: list,
    before: int = 1000, after: int = 1000, regionbody: int = 1000,
    bins: int = 100,
    split: bool = False,
    min_counts: int = 1,
    chrom_prefix: str = '',
    exclude_chr = None,
    threads=64):
    '''
    Reference-point refers to a position within a BED region (e.g., the starting point). In this mode, only those genomicpositions before (upstream) and/or after (downstream) of the reference point will be used.

    Args:
        infile: path to BAM file
        site_info: [(chrom, site1, site2, strand, gene_id), ...]
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        regionbody: distance in bases to which all regions will be fit
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        split: split mode
        min_counts: minimum number of the reads
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    chrom = site_info[:, 0]
    site1 = site_info[:, 1]
    site2 = site_info[:, 2]
    strand = site_info[:, 3]
    gene_id = site_info[:, 4]

    with ProcessPoolExecutor(max_workers=threads) as e:
        chunksize = int(len(site_info)/threads)
        results = e.map(
            bam_scale_cov, 
            repeat(infile),
            chrom,
            site1,
            site2,
            strand,
            gene_id,
            repeat(before),
            repeat(after),
            repeat(regionbody),
            repeat(bins),
            repeat(split),
            repeat(min_counts),
            repeat(chrom_prefix),
            repeat(exclude_chr),
            chunksize=chunksize)
    

    if split:
        cov_5, cov_3 = [], []
        for res in results:
            if res is not None:
                cov_5_, cov_3_ = res
                cov_5.append(cov_5_)
                cov_3.append(cov_3_)

        cov_5 = np.nanmean(cov_5, axis=0) / bins
        cov_3 = np.nanmean(cov_3, axis=0) / bins
        
        return cov_5, cov_3

    else:
        cov = []
        for res in results:
            if res is not None:
                cov.append(res)
        
        cov = np.nanmean(cov, axis=0) / bins
        return cov


def get_bam_total_readcounts(infile: str):
    """
    This function takes a bam file and returns the total number of reads in the file.
    
    Args:
      infile (str): the input bam file
    """

    return eval('+'.join([line.split('\t')[2] for line in pysam.idxstats(infile).rstrip().split('\n')]))



def plot(ax, cov, n, before=2000, after=2000, target_site=0, label=None, ylabel=None):
    """
    画metaplot
    """
    ax.plot(cov/n, label=label)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ax.set_xticks([0, before, before+after])
    ax.set_xticklabels([f'-{before//1000} Kb', target_site, f'{after//1000} Kb'], rotation=90)
    
    ax.axvline(before, ls='--', color='#555555')
    if label is not None:
        ax.legend(frameon=False)


def set_ax(
    ax, 
    bins,
    b1: int = None, a1: int = None, 
    b2: int = None, a2: int = None, 
    site1: str = 0, site2: str = 0,
    ylabel=None
    ):
    """
    This function takes in a matplotlib axis object, a list of bins, and two sets of bin numbers and
    site names, and returns a histogram of the bins with the two sets of bins highlighted
    
    Args:
      ax: the axis to plot on
      bins: the bins for the histogram
      b1 (int): int = None, a1: int = None,
      a1 (int): int = None, b1: int = None, a2: int = None, b2: int = None,
      b2 (int): int = None, a2: int = None,
      a2 (int): int = None,
      site1 (str): str = 0, site2: str = 0,. Defaults to 0
      site2 (str): str = 0,. Defaults to 0
      ylabel: str
    """
    if type(ax) is not np.ndarray:
        ax = [ax]
    
    if b1 is not None and a1 is not None:
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['top'].set_visible(False)
        ax[0].set_ylabel(ylabel)
        ax[0].set_xticks([0, b1//bins, (a1+b1)//bins])
        ax[0].set_xticklabels([f'-{b1//1000} kb', site1, f'{a1//1000} kb'], rotation=90)
        ax[0].axvline(b1//bins, ls='--', color='#555555')
    
    if b2 is not None and a2 is not None:
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[1].yaxis.set_ticks_position('none')
        ax[1].set_xticks([0, b2//bins, (a2+b2)//bins])
        ax[1].set_xticklabels([f'-{b2//1000} kb', site2, f'{a2//1000} kb'], rotation=90)
        ax[1].axvline(b2//bins, ls='--', color='#555555')


def metaplot_cen(
    infile: str, 
    cen_region: dict,
    extend: int = 100_000,
    flank_bin: int = 20,  # number of bins to use for the flanking regions
    cen_bin: int = 20,  # number of bins to use for the centromere region
    ignore_nan: bool = False,
    blacklist: dict = None,
    canonical: int = 11,
    modification: int = 12,
    threads=64):

    flank_bin += 1
    cen_bin += 1

    results = []
    for chrom in cen_region:
        
        start, end = cen_region[chrom]
        cov5_pos = np.linspace(start-extend-1, start-1, flank_bin, dtype=int)

        cov5_mask = np.zeros(len(cov5_pos), dtype=bool)
        if blacklist is not None and chrom in blacklist:
            for b in blacklist[chrom]:
                cov5_mask[np.logical_and(cov5_pos > b[0], cov5_pos < b[1])] = True

        cov3_pos = np.linspace(end, end+extend-1, flank_bin, dtype=int)
        cen_pos = np.linspace(start, end-1, cen_bin, dtype=int)

        with ProcessPoolExecutor(max_workers=threads) as e:
            chunksize = math.ceil((len(cov5_pos)-1) / threads)
            res = e.map(
                get_methyratio_within_bin, repeat(infile), repeat(chrom), cov5_pos[ :-1], cov5_pos[1: ]-1, repeat(canonical), repeat(modification), chunksize=chunksize)

        cov5 = np.array(list(res))
        cov5[cov5_mask[:-1]] = np.nan  # mask blastlist regions
        cov5 = list(cov5)

        with ProcessPoolExecutor(max_workers=threads) as e:
            chunksize = math.ceil((len(cov3_pos)-1) / threads)
            res = e.map(
                get_methyratio_within_bin, repeat(infile), repeat(chrom), cov3_pos[ :-1], cov3_pos[1: ]-1, repeat(canonical), repeat(modification), chunksize=chunksize)
        cov3 = list(res)

        with ProcessPoolExecutor(max_workers=threads) as e:
            chunksize = math.ceil((len(cen_pos)-1) / threads)
            res = e.map(
                get_methyratio_within_bin, repeat(infile), repeat(chrom), cen_pos[ :-1], cen_pos[1: ]-1, repeat(canonical), repeat(modification), chunksize=chunksize)
        cen = list(res)

        res = np.array(cov5 + cen + cov3)
        if ignore_nan:
            res[res == None] = np.nan
            
        results.append(res)
    
    # return results
    cov = np.nanmean(results, axis=0)
    return cov


# scale region
def bw_scale_cov2(
    infile: str, 
    chrom: str, site1: int, site2: int, strand: str,
    before: int = 1000, after : int = 1000, regionbody : int = 1000, 
    binsize: int = 100,
    chrom_prefix: str = '',
    normalized: str = 'density',
    exclude_chr = None,
    min_coverage: float = 0.0,
    ):
    '''
    Args:
        infile: path to tabix file
        chrom: chromosome name
        site1: 5' site
        site2: 3' site
        strand: '+' or '-'
        before: distance upstream of the site1 selected
        after: distance downstream of the site2 selected
        regionbody: distance in bases to which all regions will be fit
        bins: length in bases, of the non-overlapping bins for averaging the score over the regions length
        chrom_prefix: prefix of the chromosome name, eg. "chr"
        normalized: normalization method, 'density' or 'count'
        exclude_chr: chromosomes to be excluded
    
    Return:
        cov: the coverage value of givin regions
    '''
    site1 = int(site1)
    site2 = int(site2)
    chrom = str(chrom)

    if exclude_chr is not None and chrom in exclude_chr:
        return

    chrom = chrom_prefix + chrom

    bwfile = pysam.open(infile)
    if strand == '+':
        start = site1 - before
        end = site2 + after
    else:
        start = site1 - after
        end = site2 + before
    if start < 0 or end > bwfile.chroms()[chrom]:
        # remove Invalid interval
        return

    if strand == '+':
        start = site1 - before
        end = site2 + after
    else:
        start = site1 - after
        end = site2 + before
    
    if start < 0 or end > bwfile.chroms()[chrom]:
        return

    if strand == '+':
        cov_5 = bwfile.values(chrom, start, site1)
        cov_5 = get_bin_cov(cov_5, binsize)
        cov_3 = bwfile.values(chrom, site2, end)
        cov_3 = get_bin_cov(cov_3, binsize)
        # gene_body_region
        cov_gb = bwfile.values(chrom, site1, site2)
        cov_gb = scipy.ndimage.zoom(
            cov_gb,
            regionbody / len(cov_gb),
            order=0,
            mode='nearest')
        cov_gb = get_bin_cov(cov_gb, binsize)

    elif strand == '-':
        cov_5 = bwfile.values(chrom, site2, end)[::-1]
        cov_5 = get_bin_cov(cov_5, binsize)
        cov_3 = bwfile.values(chrom, start, site1)[::-1]
        cov_3 = get_bin_cov(cov_3, binsize)
        # gene_body_region
        cov_gb = bwfile.values(chrom, site1, site2)[::-1]
        cov_gb = scipy.ndimage.zoom(
            cov_gb,
            regionbody / len(cov_gb),
            order=0,
            mode='nearest')
        cov_gb = get_bin_cov(cov_gb, binsize)
    else:
        raise ValueError('strand must be "-" or "+"')

    cov = np.concatenate([cov_5, cov_gb, cov_3])
    # cov = np.nan_to_num(cov)

    if np.nansum(cov) > min_coverage:
        if normalized == 'density':
            cov = cov / np.nansum(cov)  # density
        return cov