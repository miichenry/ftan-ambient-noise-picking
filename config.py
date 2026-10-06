"""
Configuration for the FTAN Rayleigh-wave group-velocity picking workflow.

Adapt the paths and parameters below to your project before running.
"""
import os

# ============================================================================
# PATHS  -- edit these for each project
# ============================================================================

# Root of the project on scratch
PROJECT_ROOT = '/srv/beegfs/scratch/users/h/henrymi/project/GSE'

# Directory containing stacked cross-correlation H5 files (one subdirectory per
# source station, each containing .h5 files for every pair).
CC_PATH = os.path.join(PROJECT_ROOT, 'noisepy/GSE_STACK/CFF_phase_only_rma/linear')

# StationXML file with coordinates for all stations
STATIONXML = os.path.join(PROJECT_ROOT, 'metadata/SS_GSE_all_stations.xml')

# Output directories (created automatically)
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'post/picking_ftan')
OUTPUT_DISP = os.path.join(OUTPUT_ROOT, 'dispersion_csv')
OUTPUT_FIGS = os.path.join(OUTPUT_ROOT, 'ftan_figures')
OUTPUT_MERGED = os.path.join(OUTPUT_ROOT, 'merged')

# Network code used in the filenames (e.g. "SS.19237_SS.24184.h5")
NETWORK = 'SS'

# Stack type inside the H5 file (key under AuxiliaryData/)
STACK_TYPE = 'Allstack_linear'

# ============================================================================
# FTAN PARAMETERS
# ============================================================================

# Period range (seconds) for the filter bank
# NOTE: For the GSE array (max inter-station distance ~2.5 km), periods above
# ~2-3s violate the far-field criterion for most pairs, so TMAX=5s is appropriate.
TMIN = 0.2
TMAX = 5.0

# Group-velocity axis (km/s)
VG_MIN = 0.5
VG_MAX = 4.0
N_VG = 500

# Gaussian filter width parameters (alpha1 for basic FTAN, alpha2 for floating)
ALPHA = 5.0
ALPHA2 = 10.0

# Number of log-spaced period centres in the filter bank
N_PERIOD = 120

# ============================================================================
# PICKING / QUALITY CONTROL
# ============================================================================

# Minimum (globally-normalized) FTAN amplitude to accept a pick
PICK_THRESHOLD = 0.3

# Far-field criterion: keep picks where distance >= FAR_FIELD_FACTOR * wavelength
FAR_FIELD_FACTOR = 1.0

# Minimum per-period SNR in dB
SNR_THRESHOLD_DB = 5.0

# Components to process for FTAN (Rayleigh-wave relevant)
COMPONENTS = ['ZZ', 'RR', 'ZR', 'RZ']

# Lag types: 'sym' (symmetrized), 'pos' (causal), 'neg' (acausal)
LAG_TYPES = ['sym']

# ============================================================================
# MERGE / HISTOGRAM PARAMETERS
# ============================================================================

# SNR threshold for merging step
MERGE_SNR_THRESH = 5.0

# Distance/wavelength ratio threshold
MERGE_DLAMBDA_THRESH = 1.0
