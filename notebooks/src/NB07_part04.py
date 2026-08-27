"""Public source fragment for NB07_PRACTICAL_APPLICABILITY.ipynb.
Generated from the frozen analysis notebook; outputs and packaging-only cells excluded.
Execute fragments in numerical order within the same Python namespace.
"""

# ---- Original notebook code cell 24 ----

# PAQUETE ESTANDARIZADO NB07 — ZIP automático.
from google.colab import files as colab_files

NB_CODE = 'NB07_PRACTICAL_APPLICABILITY'
ZIP_NAME = 'NB07_RESULTS_PRACTICAL_APPLICABILITY.zip'
PACKAGE_DIR = PROJECT_ROOT / '05_RESULTS' / '_PACKAGE_TMP' / NB_CODE

if PACKAGE_DIR.exists():
    shutil.rmtree(PACKAGE_DIR)
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

shutil.copytree(RESULT_DIR, PACKAGE_DIR / '05_RESULTS_NB07')
shutil.copytree(FIG_DIR, PACKAGE_DIR / '06_FIGURES_NB07')

# Evidencia congelada mínima. El RAW no se incluye.
src = PACKAGE_DIR / 'SOURCE_FROZEN_INPUTS'
src.mkdir(parents=True, exist_ok=True)

source_files = [
    NB03_OOF, NB03_SELECTED, NB03_PROTOCOL, NB03_SUMMARY,
    NB04_SEEDWISE, NB04_SELECTED, NB04_PROTOCOL, NB04_SUMMARY,
    NB06_DESC, NB06_PAIRWISE, NB06_ORDER, NB06_PROTOCOL, NB06_SUMMARY, NB06_STATUS
]
if NB04_COMPLEXITY.exists():
    source_files.append(NB04_COMPLEXITY)
if NB04_TRAIN_METRICS.exists():
    source_files.append(NB04_TRAIN_METRICS)

for p in source_files:
    shutil.copy2(p, src / f'{p.parent.name}__{p.name}')

notebook_file = NOTEBOOKS_DIR / NOTEBOOK_FILENAME
assert notebook_file.exists(), (
    f'No se encontró {notebook_file}. Guarde este notebook en 04_NOTEBOOKS antes de empaquetar.'
)
notebook_text = notebook_file.read_text(encoding='utf-8', errors='ignore')
assert RUN_REVISION in notebook_text
notebook_sha = sha256_file(notebook_file)
shutil.copy2(notebook_file, PACKAGE_DIR / NOTEBOOK_FILENAME)

env_lines = [f'Python: {sys.version}', f'Platform: {platform.platform()}', '']
try:
    env_lines.append(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True))
except Exception as e:
    env_lines.append(f'pip freeze failed: {e}')
(PACKAGE_DIR / 'environment_packages.txt').write_text('\n'.join(env_lines),encoding='utf-8')

manifest = {
    'project': 'NIR_HUEVOS_PAPER_REBUILD_2026',
    'notebook': NB_CODE,
    'notebook_filename': NOTEBOOK_FILENAME,
    'executed_notebook_source_sha256': notebook_sha,
    'run_revision': RUN_REVISION,
    'package_schema': PACKAGE_SCHEMA,
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'raw_dataset_included_in_zip': False,
    'raw_dataset_sha256': EXPECTED_DATASET_SHA256,
    'frozen_split_manifest_sha256': EXPECTED_SPLIT_MANIFEST_SHA256,
    'generalization_source': 'frozen OOF only',
    'engineering_full_data_refit_used_for_performance': False,
    'external_validation': False,
    'zip_name': ZIP_NAME
}
(PACKAGE_DIR / 'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

readme = [
    'NB07 PRACTICAL APPLICABILITY — NIR-HUEVOS 2026',
    '',
    'STATUS: COMPLETED',
    f'RUN REVISION: {RUN_REVISION}',
    '',
    'Generalization metrics come only from frozen egg-disjoint OOF predictions.',
    'PLSR/SVR full-data refits are used only for engineering latency/size descriptors.',
    'Deep-learning latency is an architecture-level CPU reference benchmark.',
    'Primary OOF metrics are not clipped.',
    'Clipping [0,21] is secondary operational sensitivity only.',
    'No external validation and no industrial/mobile deployment claim.',
    'Raw dataset intentionally excluded from ZIP.'
]
(PACKAGE_DIR / 'README.txt').write_text('\n'.join(readme),encoding='utf-8')

inventory = []
for p in sorted(PACKAGE_DIR.rglob('*')):
    if p.is_file() and p.name != 'PACKAGE_INVENTORY.json':
        inventory.append({
            'relative_path': str(p.relative_to(PACKAGE_DIR)),
            'size_bytes': p.stat().st_size,
            'sha256': sha256_file(p)
        })
(PACKAGE_DIR / 'PACKAGE_INVENTORY.json').write_text(
    json.dumps(inventory,indent=2),encoding='utf-8'
)

zip_path = ZIP_DIR / ZIP_NAME
if zip_path.exists():
    zip_path.unlink()
shutil.make_archive(str(zip_path.with_suffix('')), 'zip', root_dir=PACKAGE_DIR)

assert zip_path.exists() and zip_path.stat().st_size > 0
print('ZIP creado:', zip_path)
print('Tamaño MB:', round(zip_path.stat().st_size/1024**2,2))

colab_files.download(str(zip_path))
