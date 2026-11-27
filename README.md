# Sentinel-2 paired-overpass BRDF sampling

This program finds **four-scene groups**: an S2A/S2B L2A pair in each of two horizontally adjacent tiles from the supplied tile list. It samples locations in both tile matchups and writes one Parquet dataset per group. No L1C correction is performed.

## Setup and SLURM

Use a Linux environment with Python 3.10 or later, GDAL with JPEG2000 support, and the packages in [requirements.txt](requirements.txt). For example, use your site's environment management or install the requirements into an existing virtual environment. Rasterio must be able to read the archive's `.jp2` files and your WorldCover mosaic/VRT.

For the supplied local Windows test, `.venv` has been created with the requirements installed. From PowerShell in this directory, run:

```powershell
.\.venv\Scripts\python.exe -m brdf --config config.local-test.yaml --start 2025-03-18 --end 2025-03-19
```

This configuration reads the four flat SAFE directories in `C:/tmp/test_data` and the local `test_worldcover_mosaic.tif` made from the two supplied WorldCover rasters. The supplied land-cover rasters are on a 20 m grid. Results are written to `test_output`. The virtual environment is local to this computer; create a separate environment on the cluster.

1. Copy `config.example.yaml` to `config.yaml`; set the L2A archive, WorldCover 2021 VRT or raster, tile-list and output paths. The archive path is the directory containing the two-digit UTM-zone directories. A tile list can be YAML (`- 32UQU`) or one tile ID per line.
2. Review the selection, sampling, geometry-band and worker settings. The example requests 200 samples per class **per tile matchup**, using a reproducible seed. With `split_cropland_ndvi: true`, it requests 200 cropland samples below the NDVI threshold and 200 at or above it per tile matchup, in addition to 200 per other class. Fewer are returned if fewer valid locations exist. `workers` is the number of groups processed concurrently and should fit the SLURM CPU and memory allocation. Each worker reads 512×512 windows by default. Begin with one worker and adjust after measuring archive I/O and memory.
3. Add the site-specific `#SBATCH` header and environment activation to `run_slurm.sh`. Submit from this repository directory, for example:

   ```bash
   sbatch --export=ALL,START=2025-04-01,END=2025-05-01,CONFIG=config.yaml run_slurm.sh
   ```

   Or run directly from the repository directory:

   ```bash
   python -m brdf --config config.yaml --start 2025-04-01 --end 2025-05-01
   ```

Dates are UTC: start is inclusive and end is exclusive. Discovery includes a time margin on either side so a group crossing midnight or a month boundary is assigned to the job containing its earliest sensing time. The program exits nonzero when discovery or processing reports errors. Inspect `output/runs/<run-id>/run.json` and any group `failure.json` as well as the SLURM log.

## Matching and selection

* Each of the four scenes must have tile-metadata `CLOUDY_PIXEL_PERCENTAGE` strictly below 20% by default. The latest baseline/production version of a given acquisition is selected **before** cloud screening. Malformed or missing required metadata is reported as an error.
* The program uses actual tile footprints and locally projected centers to identify horizontal neighbors. Both tiles must occur in the supplied list. The geographic adjacency test assumes nominal 100 km MGRS grid spacing and overlapping Sentinel-2 tile footprints; review the neighbor list in the run report, especially at UTM-zone boundaries, high latitudes and the antimeridian (which is not supported by this test).
* Candidate acquisitions must be S2A and S2B within `max_pair_minutes`, default 30. The same platform must have the same relative orbit in both tiles, with sensing times within `max_neighbor_sensing_minutes`, default 3. Matching considers all such combinations; inspect candidates if several acquisitions fall within these windows. Grouping does not assume that adjacent tiles necessarily have opposite scattering geometry.
* Both SCL values must be in `[4, 5]`, all six requested band values must be present, both interpolated sun zenith angles must be below 70°, and at least one interpolated viewing zenith angle must be below 4°. These tests are at the sample location. The closest-to-nadir sensor is recorded per row. A group can complete with zero samples in one or both tiles; check `quality` in the manifest.
* The program samples tree cover (10), grassland (30), and cropland (40) by default, using WorldCover 2021. These are WorldCover categories, not proof of permanent grassland, forest type, or unchanged 2025 land cover. With cropland splitting enabled, a separate reservoir samples near-nadir NDVI below 0.1 and another samples NDVI at or above 0.1; undefined NDVI is excluded for cropland. The threshold and split are configurable. NDVI for each acquisition and the sampling stratum are retained. The classes are sampled equally, so their pooled distribution is not area-weighted.

## Spatial and radiometric conventions

The common grid is the S2A B12 20 m grid for each tile. All rasters are aligned to this grid using **nearest-neighbor sampling**, including B04 and B08 at 10 m and the WorldCover layer. Consequently, the reported B04/B08/NDVI values are 10 m pixel samples at the 20 m grid location; they are not 20 m averages. SCL is 20 m. Misregistration, mixed pixels and the different spatial supports of the bands remain relevant to interpretation. The two Sentinel tile footprints can overlap: records retain tile, row and column so that a downstream analysis can handle overlap and avoid counting the same ground area twice.

For valid pixels, `reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE` uses the product XML's band-specific offsets and quantification value. Zero DN is excluded. The raw DN, corrected reflectance, NDVI and B-minus-A reflectance difference are retained; negative corrected reflectances are not clipped. Missing modern offsets are treated as errors.

Sun and view angles are interpolated from each tile's metadata at every sample. Detector view grids for the configured geometry band (B04 by default) are combined using a linear mean for zenith and circular mean for azimuth. Bilinear interpolation uses available nodes with at least 50% valid weight, with circular interpolation for azimuth. `sza`, `saa`, `vza`, and `vaa` are the unflipped metadata conventions in degrees; relative azimuth is view minus sun wrapped to [-180°, 180°). `solar_phase_*` is the angle between the ground-to-sun and ground-to-sensor directions, with 0° corresponding to the backscatter direction. `view_separation_deg` is the angle **between the two ground-to-sensor viewing directions**. The latter is sometimes informally called the phase angle between observers; it is different from solar phase. Geometry is calculated for the selected band, not separately for every spectral band.

Validate angle orientation, interpolation and a small sample of locations against the metadata and reference analysis on the actual HPC archive before using the results scientifically. L2A atmospheric correction and inter-sensor calibration can also contribute to reflectance differences.

## Outputs and restart behavior

`output/<date>_<tile1>_<tile2>_<fingerprint>/` contains:

* `samples.parquet`: one row per sampled location, with group/tile IDs, projected and geographic coordinates, both scene IDs and times, WorldCover and SCL classes, DN and reflectances, NDVI, geometry and differences. An empty dataset contains a `group_id` column only; the manifest carries the reason and counts.
* `manifest.json`: input identities and cloud percentages, processing configuration and source hashes, per-tile screening and sample counts, radiometric metadata and a checksum of the Parquet file. Count values describe sequential screening; geometry includes the near-nadir and solar-angle tests.
* `failure.json`, if processing failed. A successful retry removes it.

Outputs are written atomically; a complete matching manifest and Parquet checksum are checked before reuse. Linux advisory locks prevent jobs processing the same group concurrently, provided the shared filesystem supports `flock`. The fingerprint changes with scene metadata, configuration, code, and the WorldCover VRT/raster's path, size and modification time. Input image pixels and rasters *referenced by* a VRT are not hashed: treat archive products as immutable and change `processing.worldcover_revision` when VRT source data change. Existing completed groups for other fingerprints are retained.

Run reports are in `output/runs/`. A report includes the effective configuration, eligible scenes, tiles without eligible scenes, geographic neighbor pairs, group outcomes and errors. Eligible scenes can include the discovery margin outside the requested period.

## Checks

From the repository directory, run `python -m unittest discover -s tests -v`. The tests cover the four-scene example, strict cloud screening, multiple same-day overpasses, azimuth wrapping, radiometric metadata and recovery from a corrupt result. They use synthetic metadata; the repository does not contain real archive rasters or WorldCover data.

## Visualize results

`visualize_results.py` is independent of the processing package. Install NumPy, PyArrow and Matplotlib (included in `requirements.txt`), then point it at a results root:

```powershell
.\.venv\Scripts\python.exe visualize_results.py test_output --output test_visualizations
```

On Linux use the environment's `python visualize_results.py /path/to/results --output /path/to/plots`. It recursively finds completed, checksum-valid group manifests and their Parquet files. If the same four scenes occur in multiple completed groups, only the newest is included; the report lists skipped inputs. This prevents a rerun with different settings from silently counting the same overpass twice, but review the chosen settings if comparing alternative processing versions.

For a consistent sensor comparison, add `--definition a-b` to plot `100 × (S2A − S2B) / S2B` and use a separate output directory. The existing near-nadir reflectance threshold remains applied, and the same minimum also applies to S2B as the denominator; the report records both exclusions. The default `--definition off-near` preserves the near-nadir-normalized plot.

For each band the relative difference is `100 × (off-nadir reflectance − near-nadir reflectance) / near-nadir reflectance`. Results are separated by the near-nadir sensor and whether the off-nadir observation's solar phase is lower or higher than the near-nadir phase. The plot labels the off-nadir sensor as toward backscatter when its phase is lower, and toward forward scatter when higher; these are relative changes, not absolute forward/backscatter labels. Cropland is separated using the near-nadir NDVI: less than 0.1 is labelled bare and 0.1 or more is labelled higher NDVI. This is a threshold proxy, not proof of bare soil; change it with `--bare-ndvi-threshold`. Missing cropland NDVI is excluded. Values with near-nadir reflectance below 0.01 are excluded by default to limit unstable ratios; change this using `--min-near-reflectance`. Excluded counts by band are in `report.json`.

The figure contains box plots (median, quartiles, whiskers and mean); plotted whiskers omit outliers but the means and CSV statistics include them. `summary.csv` contains sample counts, overpass-group counts, sample standard deviation and sample variance, plus percentiles. The statistics pool samples, which may be spatially correlated and are not independent overpasses; the spread does not measure uncertainty across overpasses. Adjacent tiles can overlap. Inspect the input report, acquisition geometry and class sampling before generalizing a pooled result.
