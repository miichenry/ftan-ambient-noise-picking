#!/bin/bash
# Generate a list of all H5 files to process.
# Run this once before submitting the SLURM array job.

CC_PATH="/srv/beegfs/scratch/users/h/henrymi/project/GSE/noisepy/GSE_STACK/CFF_phase_only_one_bit/pws"
OUTFILE="/home/users/h/henrymi/ftan-ambient-noise-picking/h5_filelist.txt"

find "$CC_PATH" -name "*.h5" -type f | sort > "$OUTFILE"

NFILES=$(wc -l < "$OUTFILE")
echo "Found $NFILES H5 files. Written to $OUTFILE"
echo ""

# Suggest array range (100 files per task)
FILES_PER_TASK=100
NTASKS=$(( (NFILES + FILES_PER_TASK - 1) / FILES_PER_TASK ))
echo "Suggested SLURM array: --array=0-$((NTASKS - 1))"
echo "Update run_picking.slurm accordingly."
