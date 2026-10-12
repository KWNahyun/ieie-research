# Related-method comparison and claim boundaries

The review used primary full texts and official backbone resources. Verified bibliographic records are in references.json; no DOI/page numbers or publication venue are invented when only an arXiv record was verified.

The conditionedEMF reference uses a multi-resolution HRFormer, multistage scalarFiLM and radiation-pattern attention. It evaluates outdoor pixels on a random sample split. Its seven spatial channels include building encodings, a beam polygon and transmitter-relative priors. Our reference adapter retains the backbone/conditioning mechanisms but uses auditedB/T inputs, cyclic scalar encoding, full-raster loss and image-group splits. Decoder sigmoid and zero dropout/drop-path are declared adaptations. It is not an exact score reproduction; published MAE is not ranked against our different protocol.

BeamRMX emphasizes spatial radiation queries interacting with geometry and offers optional same-scene evidence. Our inputs are local one-dimensional patterns without verified world-frame/spatial-gain conversion, so a direct BeamRMX reproduction is not attempted. Pattern conditioning and geometry-query interaction are prior work, not our claimed contribution.

RadioUNet and RadioMapSeer establish propagation-image prediction and source supervision. The current smallCNN is a controlled paired backbone, not the officialRadioWNet. FiLM is an established affine-conditioning mechanism. HRFormer is a published multiresolution/window-attention backbone with an officialMIT code lineage.

The study's evidence contribution is controlled encoder-initialization comparisons under audited raw-image grouping, target geometry/configuration-density budgets, source/target seed replication, pattern-information controls and transparent renderer/provenance limitations. Propagation-specific superiority over genericimage pretraining is not isolated. Neither unseenphysical/geographic environments nor exact confirmeddata-saving rates are established. Future independent evaluation is necessary to strengthen those claims.

## Reference adapter deviations

- Input:shared two-channel image proxy/Tx versus reference geometry/beam/relative-spatial channels.
- Scalars:seven features including cyclicazimuth versus six rawscalar entries.
- Evaluation/loss:whole-raster normalizedL1 versus outdoor/compositeL1+MS-SSIM+focal loss.
- Split:60/12/14 imagegroups versus reference randomsample split.
- Augmentation:base-only versus reference flips/TTA; localasymmetric patternreflection is not verified.
- Training:fixed fair validation screen and fullfine-tuning versus referencewarmup/cosine/optimizer settings.
- Outputs/dropout:sigmoid/no stochasticdepth versus separately documented referenceactivation/backbone defaults.
- Replication:three targetseeds and one source seed for the expensivearchitecture comparison; mainCNN usesfive target/three source seeds.

All follow-up comparisons were fixed before newtraining, but after originaltest results were seen. They are supporting/exploratory; no newblind-test or universallysignificant claim is made.

Tilt-dependent radio-map transfer already transfers knowledge between antenna configurations under scarce target measurements (Parera etal.,2020; primary author repository and Nokia publication page verified). Our scope differs in source-task encoder initialization and audited image-group/configuration-density evaluation, not in claiming transfer learning itself. The previously listed TVT2025 DOI could not be fully verified through the publisher robot gate and is not used as an invented bibliographic record.
