"""Build a results-grounded paper only after required follow-up completes."""
import html
import json
from pathlib import Path
import re
import shutil
import sys
from data_pipeline import RESEARCH
from train_baseline import write_json


def table(headers, rows):
    return {'type': 'table', 'headers': headers, 'rows': [[str(v) for v in row] for row in rows]}


def main():
    root = RESEARCH / 'runs/followup_v1'
    assert json.loads((root / 'status.json').read_text())['status'] == 'experiments_complete'
    audit = json.loads((root / 'final_audit.json').read_text())
    assert audit['status'] == 'PASS' and audit['cells_verified'] == 152
    main = json.loads((RESEARCH / 'runs/main_v1/test_analysis.json').read_text())
    spatial = json.loads((RESEARCH / 'runs/spatial_ablation_v1/test_analysis.json').read_text())
    follow = json.loads((root / 'analysis.json').read_text())['results']
    supplemental = json.loads((root / 'supplementary/analysis.json').read_text())
    efficiency = json.loads((root / 'data_efficiency/analysis.json').read_text())
    provenance = json.loads((root / 'provenance/audit.json').read_text())
    source = json.loads((RESEARCH / 'runs/source_sensitivity_v1/analysis.json').read_text())
    references = json.loads((root / 'references.json').read_text())
    matched = json.loads((root / 'reference_matched_comparison.json').read_text())['rows']
    out = RESEARCH / 'manuscript'
    out.mkdir(exist_ok=True)
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    copies = [('budget_curve.png', RESEARCH / 'runs/main_v1/test_budget_curve.png'),
              ('efficiency.png', root / 'data_efficiency/equal_error_efficiency.png'),
              ('source_sensitivity.png', RESEARCH / 'runs/source_sensitivity_v1/validation_source_sensitivity.png'),
              ('spatial_interaction.png', RESEARCH / 'runs/spatial_ablation_v1/test_spatial_and_interaction.png'),
              ('qualitative.png', root / 'supplementary/prediction_error_base24.png')]
    for name, path in copies: shutil.copy2(path, figures / name)
    primary = next(r for r in main['budgets'] if r['budget'] == 25)
    lo, hi = primary['paired_geometry_bootstrap95']
    saving = next(r for r in efficiency['illustrative_thresholds'] if r['threshold'] == .080)
    title = 'Antenna-Conditioned Encoder Transfer for Normalized EMF Images: A Geometry-Group Study'
    authors = 'Author names and affiliations to be supplied'
    abstract = (f"We investigate whether propagation-supervised encoder initialization improves normalized electromagnetic-field image prediction when target data are limited. "
                f"A paired convolutional comparison uses identical geometry/transmitter inputs, scalar and radiation-pattern conditioning, and fresh target decoders. "
                f"An audit identifies768 base images organized into86 input-image groups; group-disjoint evaluation uses four training budgets, five target seeds and three source seeds. "
                f"Transfer reduces mean held-out image-group MAE by2.20–3.71% across the evaluated budgets. At the prespecified25% budget, scratch and transfer MAE are{primary['B_mae']:.6f} and{primary['D_mae']:.6f}; "
                f"the paired14-group bootstrap interval for their difference is[{lo:.6f},{hi:.6f}]. "
                f"Supporting experiments isolate pattern information, vary configuration density while retaining all training layouts, and examine an adapted conditionedHRFormer. "
                f"Source-budget diagnostics and spatial-only controls expose dependence on conditioning and source initialization. "
                f"Equal-error interpolation suggests data-efficiency gains but is explicitly descriptive, with censoring and uncertainty retained. "
                f"The study evaluates normalized raster images rather than physical exposure: independent outdoor semantics, absolute-unit inversion and geographic source–target independence remain unresolved.")
    sections = []

    def section(name, paragraphs, extras=()):
        sections.append({'title': name, 'content': [{'type': 'paragraph', 'text': p} for p in paragraphs] + list(extras)})

    section('I. Introduction', [
        'Dense field-map prediction can support repeated exploration of urban antenna configurations. When generating target labels is costly, a practical question is whether representations learned from a related propagation task lower prediction error at a fixed target-data budget. The answer requires a scratch comparator with the same target inputs and capacity; a conditioned model beating an unconditioned model does not isolate transfer.',
        'We study encoder transfer from normalized DPM radio-map images to a supplied directional-antenna EMF image dataset. The empirical task is intentionally restricted to the available raster representations. We do not infer physical exposure or independent geographic generalization from grayscale accuracy. A source encoder is transferred, while the target decoder and conditioning components are freshly initialized.',
        'The evidence contribution is a controlled initialization study supported by asset provenance checks, image-group isolation, nested geometry budgets, repeated source/target seeds, parameter-count-matched information controls, and a separate configuration-density axis. FiLM, cross-attention and transformer backbones are established methods; their introduction is not claimed as a contribution.',
        'We distinguish the original prespecified main comparison from follow-up analyses requested after its test results were known. Follow-up settings are fixed before their training and use validation selection, but their held-out results are supporting/exploratory rather than a new blind confirmation. All reported conditions, including adverse transfer outcomes, are retained.'
    ])
    section('II. Related Work', [
        'RadioUNet develops convolutional propagation-image estimation, and RadioMapSeer supplies simulated radio maps for learning and localization [1,2]. Our small U-Net is a controlled paired backbone, not a reproduction of the official RadioWNet. Transfer between antenna configurations was already studied for tilt-dependent radio maps [8]; transfer learning itself is therefore not new.',
        'FiLM implements feature-wise affine conditioning [3]. HRFormer maintains parallel multiresolution features with local-window attention [4]. The conditioned EMF reference combines multistage scalar FiLM with radiation-pattern attention [5]. Its spatial inputs, outdoor evaluation and random-sample split differ from the audited protocol here; we adapt its mechanisms for a within-backbone scratch/encoder-transfer comparison rather than rank our score against its published score.',
        'BeamRMX treats spatial radiation queries and geometry interactions as central to beam-map prediction [6]. Our available inputs are local one-dimensional patterns without verified world-frame gain conversion, so we do not reproduce its spatial-query protocol. Pattern conditioning and geometry-query interaction are prior work. SSIM is used as a complementary image-structure measure [7], not as physical correctness.',
        'Propagation supervision is used for source learning, but the experiment does not isolate its specificity relative to generic image pretraining or reconstruction pretraining. Such a control is a future mechanism study; the current claims concern measured initialization differences in the supplied tasks.'
    ])
    section('III. Data Audit and Problem Definition', [
        'The local target assets contain768 base samples,256 spreadsheet rows with three antenna-number configurations each, and86 raw building-image identities. Stored flips expand the array to3072 entries, but are excluded: image/numeric correspondence was verified while local asymmetric pattern reflection was not established. Raw building images and grayscale labels matched all base arrays, and scalar/pattern files were checked against spreadsheet and file conventions. Image identities and D4-equivalent inputs are grouped before splitting.',
        'The target split contains60/12/14 training/validation/test image groups and534/108/126 base samples. These are raw-building/image-geometry groups, not verified independent physical scenes. The local identity counts should not be equated with a differently defined layout count in a reference paper. The source manifest contains701 DPM maps with80 transmitters each; its local701st map has unresolved original-release provenance. Source map groups are split independently.',
        'Both domains use geometry and transmitter channels. Target geometry is an operational proxy: a pixel is one when both raw B/G channels are exactly zero. Source geometry is binary PNG nonzero. This convention is label-independent but not a recovered simulator footprint. A transmitter channel contains a floor-indexed coordinate marker; source alignment was verified against all antenna PNGs. Renderer and scale differences remain a domain gap.',
        'Scalar input consists of x/y, height, downtilt, power and cyclic azimuth, giving seven features after sine/cosine encoding. Horizontal and vertical patterns each contain360 ordered samples. Train-subset-only mean/standard-deviation statistics are applied to validation/test inputs. Target supervision is raw grayscale/255; source supervision is normalized DPM gain imagery. Their physical inverse scales are not assumed equivalent.',
        'The stored building mask equals the label-positive mask and changes across settings for most building identities, so it is not promoted to independent occupancy or outdoor ground truth. We score the full raster, retaining zero-valued pixels. No absolute EMF unit, exposure threshold or outdoor-only result is inferred.',
        f"Available metadata screening inspected{provenance['target_images_checked']} target and{provenance['source_images_checked']} source PNGs. It found no geographic PNG fields or target GIS/vector assets. All complete source polygon and antenna files contain local-coordinate lists. An exhaustive64x64 whole-frame D4 proxy screen made{provenance['pair_transform_comparisons']:,} comparisons, with maximum IoU{provenance['maximum_proxy_iou']:.4f} and no candidate above0.75. This cannot rule out crops, overlapping tiles, scale or renderer changes; geographic independence remains unknown."
    ], [table(['Domain/split', 'Image groups/maps', 'Base samples/pairs'], [['Target train',60,534],['Target validation',12,108],['Target test',14,126],['Local source total',701,56080]])])
    section('IV. Paired Encoder Transfer and Conditioning', [
        'Let x=(B,T), a denote normalized scalar features and r denote the two angular patterns. A spatial encoder E produces multiscale features, and a target decoder predicts a normalized image. Source learning minimizes whole-raster absolute error on DPM maps. Target adaptation copies only E; target decoder, output head and conditioning weights are initialized from the paired target seed. All encoder layers are then fine-tuned from update1.',
        'The CNN uses four encoder blocks with16/32/64/128 channels and a skip-connected decoder. Native256-pixel source and500-pixel target rasters are padded to the required pooling multiple and cropped after prediction. Conditioning concatenates a scalar MLP and a one-dimensional pattern encoder, projecting to bottleneck affine parameters. The transformation is F\u2032=(1+gamma(a,r))*F+beta(a,r), with zero-initialized affine output parameters. A/C are spatial-only scratch/source variants; B/D use the same conditioned architecture with scratch/source initialization.',
        'To separate pattern information from parameter count, the scalar-only controls retain the full conditioned CNN but replace normalized pattern tensors by zeros at every train/evaluation step. Scalar inputs remain available. The pattern encoder can learn a constant embedding, but cannot receive sample-specific pattern information. This is an information ablation, not removal of its architecture.',
        'The reference adapter uses a local MIT HRFormer-S backbone lineage with32/64/128/256 final branches. Scalar FiLM is inserted at four computational stages and a90-token pattern encoder feeds two8-head transformer-decoder layers at the deepest branch. Its native500 output uses a new skip decoder and sigmoid head. Shared two-channel inputs, cyclic scalars, full-raster L1, no beam/mask/TTA and zero stochastic depth are declared adaptations. We do not claim exact reproduction of the reference method or published outdoor score.'
    ])
    section('V. Experimental Protocol', [
        'The original primary endpoint is the paired test image-group macro MAE at25% geometry budget. Training budgets retain6/15/30/60 complete training groups, nested within each of five target subset/optimizer seeds. D uses three fixed1500-update source seeds; source checkpoints are selected by fixed source validation probes every250 updates. Source choice is not selected from target test results. Decoder/conditioning initialization and target subsets are paired.',
        'AdamW uses batch4, base channels16, target LR3e-4 and weight decay1e-4 for the CNN study. The initial cap is150 epochs with patience25. Any capped fit is rerun fresh with cap300 and the same patience; capped originals are preserved. The rule is equal for scratch and transfer. Fresh reruns can differ under nondeterministic CUDA/TF32 execution. Model selection minimizes validation group-macro MAE, and evaluation manifests are frozen before test inference.',
        'For each sample, MAE and RMSE are computed over the full raster; sample metrics are averaged within each image group, then groups receive equal weight. Seed-averaged per-group errors are paired for10,000 bootstrap draws across14 held-out groups. Intervals condition on the split and trained models, rather than treating pixels or reused controls as independent observations. Target/source seed variation is reported separately. Secondary and follow-up intervals are descriptive, with no multiplicity-adjusted significance claim.',
        'Follow-up scalar controls use the same four geometry budgets, five target seeds and three source seeds. The configuration-density axis retains all60 training image groups and178 spreadsheet rows, selecting one/two/three antenna-number entries per row with seeded nested rankings. It yields178/356/534 examples. Those entries can differ in transmitter position and height, so this is not a fixed-transmitter angular-settings experiment. Full-setting references reuse immutable main100% results.',
        'The reference adapter is compared at25/100% geometry budgets using three target seeds and one source seed0. Its source backbone is trained for1500 updates on the same DPM task. Scratch and transfer receive equal two-candidate validation LR screens,1e-4/3e-4, before final training. Reference input/loss/split changes and limited source replication are disclosed. Selection is validation-only, although original test results were already known when these supporting experiments were planned.'
    ])
    main_rows = []
    for r in main['budgets']:
        ssim = {a['model']:a['test_geometry_macro_ssim'] for a in supplemental['aggregates'] if a['budget']==r['budget']}
        main_rows.append([f"{r['budget']}%",f"{r['B_mae']:.6f}",f"{r['D_mae']:.6f}",f"{r['relative_reduction_percent']:.2f}%",f"{r['B_rmse']:.6f}/{r['D_rmse']:.6f}",f"{ssim['B']:.4f}/{ssim['D']:.4f}"])
    section('VI. Main Held-Out Image-Group Results', [
        f"Conditioned encoder transfer lowers mean test MAE at all four sampled budgets. At25%, B and D score{primary['B_mae']:.6f} and{primary['D_mae']:.6f}, an absolute difference{primary['B_minus_D']:.6f} and relative reduction{primary['relative_reduction_percent']:.2f}%. Its paired geometry-bootstrap interval is[{lo:.6f},{hi:.6f}], and13 of14 seed-averaged test groups have lower D error. These are conditional empirical image-group results, not independent physical/geographic evidence.",
        'Spatial-only transfer is less consistent: C has higher mean error than A at10% and100%, while improvements at25% and50% are small. The matched-source0 interaction (B-D_source0)-(A-C) is positive at all budgets, but its descriptive10% interval includes zero. Positive interactions do not establish a physical mechanism or universally beneficial conditioning.',
        'The qualitative example is selected from metadata before examining prediction errors: the lowest base ID within each of four lowest test-group IDs, at25% budget and target/source seed0. Shared normalized color scales are used for truth/predictions and absolute errors. This avoids replacing unfavorable examples with hand-selected cases.'
    ], [table(['Groups budget','B MAE','D MAE','Reduction','B/D RMSE','B/D SSIM'],main_rows),
        {'type':'figure','path':'figures/budget_curve.png','caption':'Original main means with target-seed SD; D averages individual-model errors across three source seeds, not an ensemble prediction.'},
        {'type':'figure','path':'figures/qualitative.png','caption':'Metadata-selected representative base24. Truth and A/B/C/D predictions use a common[0,1] scale; error maps use the same absolute-error scale. Tx/image proxy is shown without asserting physical occupancy.'}])
    scalar_rows = [[f"{r['budget']}%",f"{r['B_mae']:.6f}",f"{r['D_mae']:.6f}",f"{r['B_scalar_minus_full_pattern_mae']:.6f}",f"{r['D_scalar_minus_full_pattern_mae']:.6f}"] for r in follow['scalar']]
    setting_rows = [[r['settings_per_spreadsheet_row'],r['training_samples'],f"{r['B_mae']:.6f}",f"{r['D_mae']:.6f}",f"{r['relative_reduction_percent']:.2f}%"] for r in follow['settings']]
    reference_rows = [[f"{r['budget']}%",f"{r['B_mae']:.6f}",f"{r['D_mae']:.6f}",f"{r['relative_reduction_percent']:.2f}%"] for r in follow['reference']]
    matched_rows = [[f"{r['budget']}%", r['family'], f"{r['B_mae']:.6f}",
                     f"{r['D_mae']:.6f}", f"{r['relative_reduction_percent']:.2f}%"] for r in matched]
    reference_direction = ', '.join(
        f"{r['budget']}%: {'lower' if r['D_mae'] < r['B_mae'] else 'higher'} transfer MAE "
        f"({r['relative_reduction_percent']:.2f}% relative reduction)" for r in follow['reference'])
    pattern_benefit = sum(r[key] > 0 for r in follow['scalar']
                         for key in ['B_scalar_minus_full_pattern_mae', 'D_scalar_minus_full_pattern_mae'])
    section('VII. Supporting Information and Configuration Studies', [
        'TableIII reports zero-pattern scalar-only controls. Scalar-minus-full MAE is positive when sample-specific pattern information improves accuracy. Negative differences are retained as evidence of a nonuniform information benefit; the comparison keeps architecture/parameter count fixed and does not isolate every optimization or causal factor.',
        'TableIV varies configuration density at fixed training image-group coverage. Its100% references are reused main checkpoints, not extra independent replicates. Transmitter coordinates/heights vary across selected antenna-number entries, limiting interpretation to configuration/sample density.',
        'TableV gives the adapted HRFormer scratch/source comparison. Within-family initialization comparisons are interpretable under shared inputs and validation budgets. Published reference outdoor/random-split results are not directly ranked against these full-raster/image-group results. Limited source replication and different model capacities prevent universal architectural claims.'
        ,f'Full-pattern inputs lower mean MAE in {pattern_benefit} of eight scalar-versus-full budget/model comparisons. The adapted reference results are {reference_direction}. These outcomes qualify the scope of the original conditioned-CNN result.'
        ,'The cross-backbone table restricts both families to target seeds 0/1/2 and source seed 0. It is descriptive: model capacity, conditioning mechanisms and selected learning rates differ, so it does not isolate the causal effect of the backbone.'
    ], [table(['Group budget','Scalar B','Scalar D','B scalar-full','D scalar-full'],scalar_rows),
        table(['Entries/row','Samples','B MAE','D MAE','Reduction'],setting_rows),
        table(['Group budget','HRT scratch','HRT transfer','Reduction'],reference_rows),
        table(['Group budget','Family','Scratch MAE','Transfer MAE','Reduction'],matched_rows)])
    source_rows=[]
    for steps,seed in [(500,0),(1500,0),(4500,0),(1500,1),(1500,2)]:
        a={v['budget']:v['B_reduction_percent'] for v in source['aggregates'] if v['steps']==steps and v['source_seed']==seed}
        source_rows.append([steps,seed,f"{a[25]:.2f}%",f"{a[100]:.2f}%"])
    section('VIII. Source Sensitivity and Descriptive Data Efficiency', [
        'Source budgets500/1500/4500 are compared for seed0, while source seeds0/1/2 are compared at1500 updates. All30 paired validation comparisons favor D over shared B controls, but they are validation-selected and are not independent control replications. This is not a fully crossed source-budget-by-seed design. Source selected updates, checkpoint hashes and all target histories are retained.',
        f"Equal-error analysis interpolates the four observed mean curves in geometry-group count, using a cumulative-minimum monotonic envelope and no extrapolation. At normalized MAE0.080, point estimates require{saving['B']['required_groups_interpolated']:.2f} scratch groups versus{saving['D']['required_groups_interpolated']:.2f} transfer groups, corresponding to{saving['relative_group_savings_percent']:.2f}% interpolated savings. These fractional counts were not actually trained, and thresholds were chosen after the original test, so this is descriptive rather than a confirmed data-saving result.",
        f"Only{saving['bootstrap_joint_identifiable_fraction']:.2%} of paired bootstrap draws identify both requirements inside sampled budgets at that threshold. Censored draws are counted rather than hidden. The reported percentile interval conditions on joint identifiability and is not an unconditional confidence interval. Fresh prespecified equal-error experiments and independent evaluation are required for a stronger claim. Source pretraining cost is additional to target labeling cost."
    ], [table(['Source updates','Source seed','25% val reduction','100% val reduction'],source_rows),
        {'type':'figure','path':'figures/efficiency.png','caption':'Four actual budget points and a post-hoc equal-error interpolation curve. Interpolated savings are descriptive; neither extrapolation nor exact empirical savings are asserted.'}])
    compute_rows=[]
    for mode in ['A','B','C','D']:
        r=supplemental['compute'][mode]
        compute_rows.append([mode,r['parameters'],f"{r['conv_matmul_flops']/1e9:.3f}",f"{r['batch1_median_latency_ms']:.3f}",f"{r['inference_peak_allocated_bytes']/2**20:.1f}"])
    section('IX. Computational Cost and Reproducibility', [
        'TableVII reports model-only CUDA-event timing after50 warmups and200 timed iterations, with inputs resident on anRTX4090 GPU. Current shared-GPU load is disclosed. FLOPs include supported convolution/matrix operations and exclude unsupported normalization/activation/interpolation operations; they are not total-algorithm FLOPs. Inference memory uses the process allocator, and synthetic batch4 forward/backward memory is not a historical training peak.',
        'Recorded source-pretraining and target-adaptation stage runtimes are stored separately, including capped originals and fresh retries. Their wall-time sum is not GPU-active time or end-to-end time. Saved protocol/code/input hashes, group/subset IDs, train normalization, checkpoint selection, per-sample/group metrics, environment versions and failure/cap provenance support reproduction.',
        'Independent audits verify frozen checkpoints, finite strict inference, train-only statistics, group isolation, exact evaluation topology and recalculated MAE/RMSE/SSIM. The original120 main/spatial checkpoints and all152 supporting cells are retained. Test values are not used to retrain or reselect checkpoints. Physical unknowns and supporting-test status remain part of the reproducibility record.'
    ], [table(['Model','Parameters','Conv/mm GFLOPs','Median ms','Peak alloc MiB'],compute_rows)])
    section('X. Discussion and Limitations', [
        'The measurements support an empirical benefit from conditioned source-encoder initialization on these normalized-image tasks. The spatial-only mixed results and information controls caution against treating pretrained weights or additional antenna inputs as universally beneficial. Encoder transfer and conditioning are evaluated through paired predictors; favorable D-versus-C accuracy alone is not transfer evidence.',
        'The data do not establish verified independent geographic scenes. Scene IDs, tile extents, geographic source-target overlap, independent outdoor/nodata semantics and physical inverse normalization are unavailable. Coarse image nonmatches cannot exclude overlapping crops or different renderers. Image-group bootstrap intervals omit these uncertainties and do not establish real-world exposure accuracy.',
        'The source-budget study is sparse across seeds, and the expensive reference architecture uses a single source seed. Follow-up design occurs after the original test was seen. Its results are transparently supporting rather than new blind confirmation. Generic-image pretraining, additional simulators, frozen encoders and exact fixed-transmitter angular settings remain future studies, not completed evidence.',
        'A larger independent dataset with recoverable scene/geometry metadata and physical labels would permit stronger generalization and exposure claims. A fresh prespecified equal-error study could test interpolated labeling savings. Such work should retain unfavorable outcomes and avoid adapting hyperparameters to the existing held-out errors.'
    ])
    section('XI. Conclusion', [
        f"A paired, audited image-group study finds lower conditioned-CNN test MAE from source encoder transfer across four target geometry budgets, with{primary['relative_reduction_percent']:.2f}% reduction at the prespecified25% endpoint. Scalar information controls, configuration-density comparisons and an adapted reference backbone provide supporting evidence about the scope of the effect. Data-efficiency interpolation is descriptive, while physical/geographic independence and absolute-unit interpretation remain unresolved. The appropriate conclusion is a bounded normalized-image transfer result rather than an exposure prediction or universal data-saving guarantee."
    ])
    document = {'title':title,'authors':authors,'abstract':abstract,'keywords':'encoder transfer; antenna conditioning; normalized EMF images; geometry-group evaluation; data efficiency',
                'sections':sections,'references':references,'status':'research draft; author/affiliation/venue details pending; not submitted',
                'supporting_test_scope':'follow-up original test previously seen','source_audit':audit}
    def readable(text):
        text = re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])', ' ', text)
        text = re.sub(r'(?<=%)(?=[A-Za-z])', ' ', text)
        text = text.replace('are[', 'are [').replace('is[', 'is [')
        for old, new in [('TableIII', 'The scalar-control table'),
                         ('TableIV', 'The configuration-density table'),
                         ('TableV', 'The reference-backbone table'),
                         ('TableVII', 'The computational-cost table')]:
            text = text.replace(old, new)
        for old, new in [('conditionedHRFormer', 'conditioned HRFormer'), ('RadioWNet', 'RadioWNet'),
                         ('imagedata', 'image data'), ('targetdata', 'target data')]:
            text = text.replace(old, new)
        return text
    document['abstract'] = abstract = readable(abstract)
    for s in sections:
        for block in s['content']:
            if block['type'] == 'paragraph': block['text'] = readable(block['text'])
            if block['type'] == 'figure': block['caption'] = readable(block['caption'])
    write_json(out / 'manuscript_content.json', document)
    md=[f'# {title}','',authors,'','## Abstract','',abstract,'','Keywords: '+document['keywords'],'']
    for s in sections:
        md += ['## '+s['title'],'']
        for block in s['content']:
            if block['type']=='paragraph': md += [block['text'],'']
            elif block['type']=='table':
                md += ['| '+' | '.join(block['headers'])+' |','| '+' | '.join(['---']*len(block['headers']))+' |']
                md += ['| '+' | '.join(row)+' |' for row in block['rows']];md += ['']
            else:md += [f"![{block['caption']}]({block['path']})",'',block['caption'],'']
    md += ['## References','']
    for i,ref in enumerate(references,1):md += [f"[{i}] {ref['authors']}, {ref['title']}, {ref.get('venue','arXiv preprint')}, {ref['year']}. {ref['url']}",'']
    (out / 'manuscript.md').write_text('\n'.join(md))
    render_docx(out,document)
    render_pdf(out,document)
    render_tex(out,document)
    print('Results-grounded paper generated:Markdown,DOCX,reviewPDF,LaTeX source')


def imports():
    sys.path.insert(0,str(RESEARCH/'vendor/manuscript_deps'))


def render_docx(out,document):
    imports()
    from docx import Document
    from docx.shared import Inches,Pt
    doc=Document();style=doc.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(10)
    doc.add_heading(document['title'],0);doc.add_paragraph(document['authors']);doc.add_paragraph(document['status'])
    doc.add_heading('Abstract',1);doc.add_paragraph(document['abstract']);doc.add_paragraph('Keywords: '+document['keywords'])
    for section in document['sections']:
        doc.add_heading(section['title'],1)
        for b in section['content']:
            if b['type']=='paragraph':doc.add_paragraph(b['text'])
            elif b['type']=='table':
                t=doc.add_table(rows=1,cols=len(b['headers']));t.style='Table Grid'
                for cell,text in zip(t.rows[0].cells,b['headers']):cell.text=text
                for row in b['rows']:
                    for cell,text in zip(t.add_row().cells,row):cell.text=text
            else:doc.add_picture(str(out/b['path']),width=Inches(6.4));doc.add_paragraph(b['caption'])
    doc.add_heading('References',1)
    for i,r in enumerate(document['references'],1):doc.add_paragraph(f"[{i}] {r['authors']}. {r['title']}. {r.get('venue','arXiv')}, {r['year']}. {r['url']}")
    doc.save(out/'manuscript.docx')


def render_pdf(out,document):
    imports()
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.utils import ImageReader
    styles=getSampleStyleSheet();styles.add(ParagraphStyle('PaperBody',fontName='Times-Roman',fontSize=10,leading=12,alignment=TA_JUSTIFY,spaceAfter=7))
    styles.add(ParagraphStyle('PaperCaption',fontName='Times-Italic',fontSize=8,leading=10,spaceAfter=8))
    pdf=SimpleDocTemplate(str(out/'manuscript_review.pdf'),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=45,bottomMargin=45,title=document['title'],author='Author details pending')
    clean=lambda t:html.escape(t.replace('\u2032',"'").replace('–','-').replace('×','x'))
    story=[Paragraph(clean(document['title']),styles['Title']),Paragraph(clean(document['authors']),styles['Normal']),Paragraph(clean(document['status']),styles['PaperCaption']),Spacer(1,12),Paragraph('Abstract',styles['Heading2']),Paragraph(clean(document['abstract']),styles['PaperBody'])]
    for section in document['sections']:
        story.append(Paragraph(clean(section['title']),styles['Heading2']))
        for b in section['content']:
            if b['type']=='paragraph':story.append(Paragraph(clean(b['text']),styles['PaperBody']))
            elif b['type']=='table':
                cells=[[Paragraph(clean(x),styles['PaperCaption']) for x in row] for row in [b['headers']]+b['rows']]
                t=Table(cells,colWidths=[516/len(b['headers'])]*len(b['headers']),repeatRows=1)
                t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eeeeee')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4)]));story += [t,Spacer(1,10)]
            else:
                path=out/b['path'];w,h=ImageReader(str(path)).getSize();story += [Image(str(path),width=516,height=516*h/w),Paragraph(clean(b['caption']),styles['PaperCaption'])]
    story.append(Paragraph('References',styles['Heading2']))
    for i,r in enumerate(document['references'],1):story.append(Paragraph(clean(f"[{i}] {r['authors']}. {r['title']}. {r.get('venue','arXiv')}, {r['year']}. {r['url']}"),styles['PaperCaption']))
    def page(canvas,doc):
        canvas.setFont('Times-Roman',8);canvas.drawString(48,25,'Research draft - normalized-image evidence; venue formatting pending');canvas.drawRightString(564,25,str(doc.page))
    pdf.build(story,onFirstPage=page,onLaterPages=page)


def render_tex(out,document):
    def esc(t):
        t=t.replace('\u2032',"'").replace('–','--').replace('×',r'$\times$')
        for old,new in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:t=t.replace(old,new)
        return t
    tex=[r'\documentclass[conference]{IEEEtran}',r'\usepackage{graphicx,booktabs,amsmath,url}',r'\begin{document}',r'\title{'+esc(document['title'])+'}',r'\author{\IEEEauthorblockN{Author details pending}\IEEEauthorblockA{Affiliations pending}}',r'\maketitle',r'\begin{abstract}',esc(document['abstract']),r'\end{abstract}',r'\begin{IEEEkeywords}',esc(document['keywords']),r'\end{IEEEkeywords}']
    for s in document['sections']:
        title=re.sub(r'^[IVX]+\.\s*','',s['title']);tex += [r'\section{'+esc(title)+'}']
        for b in s['content']:
            if b['type']=='paragraph':tex += [esc(b['text']),'']
            elif b['type']=='figure':tex += [r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=0.95\textwidth]{'+b['path']+'}',r'\caption{'+esc(b['caption'])+'}',r'\end{figure*}']
            else:
                tex += [r'\begin{table*}[t]\centering\small',r'\begin{tabular}{'+'l'*len(b['headers'])+'}',r'\toprule',' & '.join(esc(x) for x in b['headers'])+r' \\',r'\midrule']
                tex += [' & '.join(esc(x) for x in row)+r' \\' for row in b['rows']];tex += [r'\bottomrule\end{tabular}\end{table*}']
    tex += [r'\begin{thebibliography}{99}']
    for i,r in enumerate(document['references'],1):tex += [r'\bibitem{'+r['key']+'} '+esc(f"{r['authors']}, {r['title']}, {r.get('venue','arXiv')}, {r['year']}.")+r' \url{'+r['url']+'}']
    tex += [r'\end{thebibliography}',r'\end{document}'];(out/'manuscript.tex').write_text('\n'.join(tex)+'\n')
    (out/'README.md').write_text('Results-grounded research draft; author/affiliation/venue details pending.\n\nmanuscript.md and.docx are editable. manuscript_review.pdf is a single-column review copy, not a compiled IEEEtran PDF. manuscript.tex usesIEEEtran; compile with a LaTeX installation containingIEEEtran (not available in this environment). Figures are in figures/. Sources and claim/provenance limits are retained. No submission or external publication performed.\n')


if __name__ == '__main__':
    main()
