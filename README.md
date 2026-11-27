# Sentinel-2 paired-overpass BRDF sampling

This program finds **four-scene groups**: an S2A/S2B L2A pair in each of two horizontally adjacent tiles from the supplied tile GeoPackage or text list. It samples locations in both tile matchups and writes one Parquet dataset per group. No L1C correction is performed.

## Setup and SLURM

Use a Linux environment with Python 3.10 or later, GDAL with JPEG2000 support, and the packages in [requirements.txt](requirements.txt). For example, use your site's environment management or install the requirements into an existing virtual environment. Rasterio must be able to read the archive's `.jp2` files and your WorldCover mosaic/VRT.

For the supplied local Windows test, `.venv` has been created with the requirements installed. From PowerShell in this directory, run:

```powershell
.\.venv\Scripts\python.exe -m brdf --config config.local-test.yaml --start 2025-03-18 --end 2025-03-19
```

This configuration reads the four flat SAFE directories in `C:/tmp/test_data` and the local `test_worldcover_mosaic.tif` made from the two supplied WorldCover rasters. The supplied land-cover rasters are on a 20 m grid. Results are written to `test_output`. The virtual environment is local to this computer; create a separate environment on the cluster.

1. Copy `config.example.yaml` to `config.yaml`; set the L2A archive, WorldCover 2021 VRT or raster, tile GeoPackage and output paths. The archive path is the directory containing the two-digit UTM-zone directories. For the tile GeoPackage, set `paths.tile_geopackage` and `tiles.tile_id_field: Tile`; specify `tiles.geopackage_layer` if it contains multiple feature layers. A GeoPackage with exactly one feature layer and exactly one field containing valid tile IDs can be detected automatically if these settings are omitted. Tile names can optionally start with `T`.
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
* With a tile GeoPackage, the program uses its feature polygon geometries (for example the `UTM_WKT` multipolygon geometry column; the input CRS may be EPSG:4326) and projects each into the tile's UTM zone. Two tiles are horizontal neighbors when they are in the same UTM zone and hemisphere and their polygon centroids differ by approximately 100 km in easting and have approximately equal northings. `tiles.neighbor_tolerance_m` defaults to 2,500 m for both distances. This supports touching 100 km cells and overlapping Sentinel-2 tile footprints, but assumes full, consistently shaped tile polygons; clipped or irregular footprints can shift centroids and cause missed or incorrect pairs. Cross-zone pairs are deliberately excluded. Review `neighbor_pairs` and `tile_geopackage` in the run report before a large run.
* A text tile list is still supported: replace `paths.tile_geopackage` with `paths.tile_list` pointing to a YAML list (`- 32UQU`) or one ID per line. With a text list, neighbors are inferred from the archive scene metadata using the earlier geographic footprint heuristic; that heuristic does not support the antimeridian. Only tiles in the supplied input are considered.
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

Run reports are in `output/runs/`. A report includes the effective configuration, eligible scenes, tiles without eligible scenes, neighbor pairs, group outcomes and errors. For a GeoPackage it also records the selected layer and field, tiles with no horizontal neighbor in the input, and pairs where both tiles have eligible scenes. Eligible scenes can include the discovery margin outside the requested period.

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

By default, `variance_per_pair.png` shows one point for each **tile matchup**, band, land-cover class and sensor/phase category. Each point is the sample variance (`ddof=1`) of the pixel-level relative differences within that subset; pixels are not pooled across tile matchups. One four-scene overpass group normally contributes two tile matchups. The plot uses a symmetric-log variance axis by default; use `--variance-scale linear` for a linear scale. `pair_summary.csv` records each matchup's tile and scene IDs, acquisition times, sample count, mean, median, quartiles, interquartile range, standard deviation and variance. Standard deviations are in percentage points and variances in squared percentage points. Variance includes outliers, is sensitive to extreme ratios, and describes within-matchup pixel spread, not uncertainty across independent overpasses.

At least two valid samples are required to calculate variance; use `--min-pair-samples` to require more. Subsets with too few samples remain in the CSV with status `too_few_samples` but are omitted from the plot. If a tile's identical S2A/S2B acquisition pair appears in several adjacent-tile overpass groups, the latest completed group is selected for that matchup and skipped duplicates are listed in `report.json`. This is separate from the four-scene-group deduplication described above. Adjacent tiles can geographically overlap; their pixel samples are not deduplicated across tile IDs. Use a new output directory when changing options so old plots cannot be mistaken for new results.

The former pooled box plot and pooled `summary.csv` can optionally be generated with `--include-pooled`. That figure has median and quartiles, whiskers and mean; the plotted whiskers omit outliers but the mean and CSV statistics include them. Pooled samples may be spatially correlated and are not independent overpasses; inspect the input report, acquisition geometry and class sampling before generalizing.

### Investigate land-cover differences

`investigate_landcover.py` is a separate analysis script. It reuses the checksum-verified result discovery in `visualize_results.py`, but does not run the sampling pipeline or change its outputs:

```bash
python investigate_landcover.py /path/to/results --output /path/to/landcover_investigation
```

It uses the consistent `100 × (S2A − S2B) / S2B` definition, and requires both S2B and the near-nadir reflectance to be at least 0.01. Use `--min-reflectance` to change the threshold and `--min-samples` to change the minimum valid samples per tile matchup, band, class/bin and sensor/phase category (default 10). Subsets below the minimum are recorded in the CSV but not plotted or used for contrasts. Each scene pair is analyzed separately; if it appears in multiple overpass groups, the latest completed instance is used and the choice is recorded.

The output includes three figures and their underlying CSV tables:

* `01_within_matchup_spread.png` and `class_summary.csv`: IQR and sample variance of the relative differences, per matchup and land-cover class. IQR is in percentage points; variance is in squared percentage points and includes outliers. The variance panels use a symmetric-log scale.
* `02_contrast_vs_geometry.png` and `geometry_contrasts.csv`: median difference for cropland with near-nadir NDVI < 0.1 minus the median of each comparison class in the **same** tile matchup, band and sensor/phase category. The horizontal coordinate is the median off-nadir minus near-nadir solar phase angle of the bare-cropland samples. The CSV also records the comparison-class geometry and both sample counts. Different classes are sampled at different locations within the tile, so their geometry can differ.
* `03_cropland_ndvi.png` and `ndvi_bin_summary.csv`: one line per matchup/band/sensor/phase category, showing the median relative difference in near-nadir NDVI bins `<0`, `[0,0.1)`, `[0.1,0.2)`, `[0.2,0.4)`, `[0.4,0.6)` and `≥0.6`. Lines do not bridge missing or insufficient bins. Stratified sample counts are not area fractions.

`report.json` lists inputs, exclusions, duplicate selections and settings. NDVI < 0.1 is a bare-soil proxy, not ground truth. NDVI and the B04/B08 response use some of the same reflectances, which can induce mathematical coupling. The figures show within-matchup patterns; samples can be spatially correlated, and a single overpass cannot establish a systematic effect.

### Map high-variance bare-cropland matchups

Run the separate mapping and drill-down script after sampling:

```bash
python map_bare_variance.py /path/to/results \
  --output /path/to/bare_variance_map \
  --band B08 --top 5 --min-samples 10 \
  --tiles-geopackage /path/to/tiles.gpkg --tile-id-field Tile
```

The GeoPackage and its field option are optional; add `--tile-layer` if it contains multiple feature layers. With no GeoPackage, the script attempts to draw outlines from the original Sentinel-2 tile metadata when those archive paths are accessible. Otherwise it still plots sample locations. The figure is a rough longitude/latitude map with tile outlines when available, not a coastline or political basemap.

`bare_variance_map.png` maps the median longitude and latitude of qualifying bare-cropland sample coordinates for each tile matchup. The color is the sample variance of `100 × (S2A − S2B)/S2B` over near-nadir NDVI < 0.1, for the selected band (B08 by default). At least `--min-samples` valid samples are required per sensor/phase category. If a scene pair has multiple such categories, its highest qualifying within-category variance determines its rank; pixels from different categories are not pooled. The ranking is relative to the available dataset, not a universal definition of high variance. Variance can be driven by extreme ratios; both S2B and near-nadir reflectance must be ≥0.01 by default, adjustable with `--min-reflectance`.

`bare_variance_ranking.csv` records scene IDs, times, tile, sensor/phase category, coordinates, sample count, variance and IQR. Different dates at the same tile may overlap on the map; use the CSV to inspect every ranked matchup. For each of the `--top` matches, `high_variance/` contains `relative_difference.png` and `summary.csv` for that **single S2A–S2B tile matchup**, plus `adjacent_tiles_overview.png` and `adjacent_tiles_summary.csv` for its original four-scene, two-tile group. Each folder's `metadata.json` identifies the source scenes and selection. Use a fresh output directory when changing the ranking options so previous high-variance folders are not mistaken for the current selection.
