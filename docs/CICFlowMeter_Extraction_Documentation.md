# CICFlowMeter Re-Extraction of CICIoMT2024
## Oswald Adohinzin — Dakota State University

*Document created: March 2026*
*Purpose: Document the PCAP → CICFlowMeter CSV re-extraction process for thesis reproducibility.*

---

## 1. Motivation

The CICIoMT2024 dataset ships with pre-extracted CSV files that use **DPKT extraction**
(39 packet-window features). However, all three cross-dataset generalization targets
(CIC-BoT-IoT, CIC-IoT-DIAD-2024, CIC-ToN-IoT) use **CICFlowMeter extraction**
(83 per-flow columns / 76 ML-usable features). This fundamental feature mismatch made
cross-dataset transfer impossible without re-extraction.

**Solution**: Re-extract all 72 CICIoMT2024 PCAP files using the same Java CICFlowMeter
tool (v4.0) used by the other datasets, ensuring 100% feature alignment.

---

## 2. Source Data

- **72 PCAP files**: 36 train + 36 test (18 attack types + Benign, each with train/test split)
- **Location on HPC**: `~/CICIoMT2024/pcaps/` (uploaded from Google Drive)
- **Original Google Drive path**: `Conf_Paper/dissertation/data/CICIoMT2024/`

---

## 3. Extraction Tool

- **Tool**: Java CICFlowMeter v4.0
- **Repository**: https://github.com/ahlashkari/CICFlowMeter
- **Build system**: Gradle 4.10.3 (NOT the bundled wrapper, which is too old for Java 11)
- **Java version**: OpenJDK 11 (Java 11.0.1 on HPC via `module load java/jdk11.0.1`)
- **Native dependency**: jnetpcap 1.4.r1425 (installed to local Maven repo)
- **System dependency**: libpcap (symlinked from `/usr/lib64/libpcap.so.1` on HPC)

### Build Notes
- The Gradle build requires downloading dependencies from Maven Central via HTTPS.
  Java 11.0.1 (from 2018) has outdated TLS certificates and fails the handshake.
- **Workaround**: Built on Google Colab (which has newer Java 11), then transferred
  the distribution zip (`CICFlowMeter-4.0-dist.zip`) to the HPC cluster.
- Gradle 7+ is NOT compatible with the project's `build.gradle` (uses deprecated
  `maven` plugin and `runtime` configuration).

### CLI Invocation
The correct way to invoke CICFlowMeter from command line:
```bash
java -Xmx28g \
  "-Djava.library.path=$NATIVE_DIR" \
  -cp "$CLASSPATH" \
  cic.cs.unb.ca.ifm.Cmd \
  "$INPUT_DIR" "$OUTPUT_DIR"
```
- The `CICFlowMeter` and `cfm` shell scripts do NOT work reliably
- CICFlowMeter requires a **directory** as input (not individual files)
- Workaround: Create a temp directory with a symlink to a single PCAP file per job

---

## 4. HPC Cluster Details

- **Cluster**: "Innovator" at South Dakota State University
- **Portal**: hpcportal.sdstate.edu (Open OnDemand)
- **OS**: Rocky Linux 9.1
- **Scheduler**: SLURM
- **Partition used**: `compute`
- **Resources per job**: 32 GB RAM, 6-hour time limit, 1 CPU

---

## 5. Extraction Process

### Step 1: Setup (one-time)
- Loaded Java 11 module, installed jnetpcap to local Maven repo
- Transferred pre-built CICFlowMeter distribution from Colab
- Created native library directory with jnetpcap `.so` files + libpcap symlink
- Generated `pcap_list.txt` listing all 72 PCAP files

### Step 2: SLURM Array Jobs
- Submitted as SLURM array job: `--array=1-72`
- Each job: creates temp dir → symlinks one PCAP → runs CICFlowMeter → adds labels
- Critical environment setup:
  ```bash
  module load java/jdk11.0.1
  export LD_LIBRARY_PATH="$NATIVE_DIR:/usr/lib64:$LD_LIBRARY_PATH"
  ```

### Step 3: Labeling
- Python script (`03_add_labels.py`) adds Label, Class, and Sub-Class columns
  based on the PCAP filename
- 29-entry LABEL_MAP covering all filename variants (underscores, hyphens, abbreviations)
- Label hierarchy: Sub-Class (individual attack) → Class (attack family) → Label (binary)

### Step 4: Verification
- All 72/72 files successfully extracted
- All files have identical 86 columns (83 CICFlowMeter + Label + Class + Sub-Class)
- Total flows: 6,732,090
- All 19 classes represented (18 attack types + Benign)

---

## 6. Output Summary

| Metric | Value |
|--------|-------|
| Files extracted | 72 / 72 |
| Total columns | 86 (83 CICFlowMeter + 3 label columns) |
| ML-usable features | 76 (excluding 7 flow identifiers) |
| Total flows | 6,732,090 |
| Classes | 19 (18 attacks + Benign) |
| Attack families | 5 (DDoS, DoS, MQTT, Recon, Spoofing) |

### Class Distribution
- Benign: 627 flows (very small — requires careful class balancing)
- Largest classes: DDoS and DoS variants (millions of flows)
- All 18 attack types + Benign present in both train and test splits

### Output File Naming Convention
```
{AttackType}_{split}.pcap_Flow.csv
```
Examples: `ARP_Spoofing_test.pcap_Flow.csv`, `Benign_train.pcap_Flow.csv`

---

## 7. File Locations

| Item | HPC Path |
|------|----------|
| Raw PCAPs | `~/CICIoMT2024/pcaps/` |
| CICFlowMeter distribution | `~/CICIoMT2024/cicfm_dist/CICFlowMeter-4.0/` |
| Raw CICFlowMeter output | `~/CICIoMT2024/csv_raw/` |
| Labeled CSVs (final) | `~/CICIoMT2024/csv_labeled/` |
| SLURM logs | `~/CICIoMT2024/logs/` |
| Setup script | `~/CICIoMT2024/01_setup.sh` |
| SLURM job script | `~/CICIoMT2024/02_slurm_extract.sh` |
| Labeling script | `~/CICIoMT2024/03_add_labels.py` |
| Verification script | `~/CICIoMT2024/04_verify.py` |

| Item | Google Drive / Local Path |
|------|--------------------------|
| Final labeled CSVs | `dissertation/data/CICIoMT2024/CSV/` |

---

## 8. Reproducibility Notes

1. **Java version matters**: Use Java 11 specifically. Java 17+ causes Gradle version incompatibilities; Java 8 lacks required features.
2. **Gradle version matters**: Use Gradle 4.10.3 specifically. The bundled wrapper is too old; Gradle 7+ breaks the build.gradle syntax.
3. **TLS certificates**: Java 11.0.1 may fail HTTPS connections to Maven Central. Build on a machine with updated certificates, then transfer the distribution.
4. **libpcap**: Must be available as `libpcap.so` (not just `libpcap.so.1`). Create a symlink if needed.
5. **Memory**: Large PCAP files (>1 GB) require 16+ GB RAM. 32 GB is safe for all files.
6. **Directory input**: CICFlowMeter only accepts directories, not individual files. Use the symlink-in-temp-dir workaround for per-file processing.

---

## 9. Key Lesson: DPKT vs CICFlowMeter

The original CICIoMT2024 CSV files use DPKT extraction, producing 39 packet-window
features. These are fundamentally different from CICFlowMeter's 83 per-flow features:

- **DPKT**: Packet-level statistics over sliding windows
- **CICFlowMeter**: Bidirectional flow-level statistics (forward + backward)

The feature names partially overlap (e.g., both have "Flow Duration"), but the semantics
and computation methods differ. Direct comparison or transfer between DPKT and CICFlowMeter
features is not valid. This is why re-extraction was mandatory for the cross-dataset
generalization contribution.
