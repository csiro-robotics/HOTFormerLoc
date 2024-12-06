import React from "react";
import Slideshow from "../../components/Slideshow/Slideshow";
import styles from "./Paper.module.css";

const images = [
  "/hotformerloc/assets/slides/karawatha_image_1.png",
  "/hotformerloc/assets/slides/karawatha_image_2.png",
  "/hotformerloc/assets/slides/karawatha_image_3.png",
];

const Paper: React.FC = () => {
  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>
          HOTFormerLoc: Hierarchical Octree Transformer for Lidar Place
          Recognition
        </h1>
      </header>
      <main className={styles.main}>
        <div className={styles.section}>
          <a>Download the full paper here!</a>
        </div>

        <section className={styles.section}>
          <Slideshow images={images} />
        </section>

        <section className={styles.section}>
          <h2 id="abstract" className={styles.sectionHeading}>
            Abstract
          </h2>
          <p className={styles.paragraph}>
            We present HOTFormerLoc, a novel and versatile Hierarchical
            Octree-based Transformer for large-scale 3D place recognition in
            both ground-to-ground and ground-to-aerial scenarios across urban
            and forest environments. Leveraging an octree-based structure, we
            propose a multi-scale attention mechanism that captures spatial and
            semantic features across granularities.
          </p>
        </section>

        <section className={styles.section}>
          <h2 id="network-architecture" className={styles.sectionHeading}>
            Network Architecture
          </h2>
          <p className={styles.paragraph}>
            We use an octree to generate a hierarchical feature pyramid F, which
            is tokenised and partitioned into local attention windows F̂l of size
            k (k = 3 in this example). We introduce a set of relay tokens RT_l
            representing local regions at each level and process both local and
            relay tokens in a series of HOTFormer blocks. A pyramid attention
            pooling layer then aggregates the multi-scale features into a single
            global descriptor.
          </p>

          <div className={styles.imageGrid}>
            <h3 id="hotformerloc" className={styles.subHeading}>
              HOTFormerLoc
            </h3>
            <figure className={styles.figure}>
              <img
                src="/hotformerloc/assets/architecture/architecture_hotformerloc.png"
                alt="HOTFormerLoc Architecture Diagram"
                className={styles.image}
              />
              <figcaption>HOTFormerLoc Architecture</figcaption>
            </figure>
          </div>

          <div>
            <h3 id="rtsa" className={styles.subHeading}>
              Relay Token Self-Attention (RTSA) Block
            </h3>
            <p className={styles.paragraph}>
              HOTFormer blocks consist of relay token self-attention (RTSA) to
              induce long-distance multi-scale interactions.
            </p>
            <div className={styles.imageGrid}>
              <figure className={styles.figure}>
                <img
                  src="/hotformerloc/assets/architecture/architecture_rtsa.png"
                  alt="RTSA Block Architecture Diagram"
                  className={styles.image}
                />
                <figcaption>RTSA Block Diagram</figcaption>
              </figure>
              <figure className={styles.figure}>
                <img
                  src="/hotformerloc/assets/architecture/architecture_rtsa_2.png"
                  alt="RTSA Attention Visualization"
                  className={styles.image}
                />
                <figcaption>
                  Relay token multi-scale attention visualized on the octree
                  feature pyramid.
                </figcaption>
              </figure>
            </div>
          </div>

          <div>
            <h3 id="hosa" className={styles.subHeading}>
              Hierarchical Octree Self-Attention (HOSA) Block
            </h3>
            <p className={styles.paragraph}>
              HOTFormer blocks also consist of hierarchical octree
              self-attention (HOSA) to refine local features and propagate
              global contextual cues learned by the relay tokens.
            </p>
            <div className={styles.imageGrid}>
              <figure className={styles.figure}>
                <img
                  src="/hotformerloc/assets/architecture/architecture_hosa.png"
                  alt="HOSA Block Architecture Diagram"
                  className={styles.image}
                />
                <figcaption>HOSA Block Diagram</figcaption>
              </figure>
            </div>
          </div>
        </section>

        <div className={styles.coaDiagram}>
          <h3 id="coa" className={styles.subHeading}>
            Cylindrical Octree Attention
          </h3>
          <p className={styles.paragraph}>
            Cartesian VS cylindrical attention window serialisation (each window
            indicated by the arrow colour) for the 2D equivalent of an octree
            with depth d = 3 and window size k = 7. Cylindrical octree attention
            windows better represent the distribution of spinning lidar point
            clouds.
          </p>
          <div className={styles.imageGrid}>
            <figure>
              <img
                src="/hotformerloc/assets/architecture/architecture_coa_2.png"
                alt="Cylindrical Octree Attention Architecture Diagram"
                className={styles.image}
              />
              <figcaption>Cylindrical Octree Attention Diagram</figcaption>
            </figure>
          </div>
        </div>

        <div>
          <h3 id="pap" className={styles.subHeading}>
            Pyramid Attention Pooling
          </h3>
          <p className={styles.paragraph}>Pyramid Attention Pooling</p>
        </div>

        <section className={styles.section}>
          <h2 id="experiments" className={styles.sectionHeading}>
            Experiments
          </h2>
          <p className={styles.paragraph}>
            This section explores the datasets and evaluation criteria used for
            our experiments, along with insights gained from ablation studies.
          </p>
          <div>
            <h3 id="evaluation-criteria" className={styles.subHeading}>
              Datasets and Evaluation Criteria
            </h3>
            <p className={styles.paragraph}>
              To demonstrate our method's versatility, we conduct experiments on
              Oxford RobotCar, CS-Campus3D, and Wild-Places, using the
              established training and testing splits for each,
            </p>

            <div className={styles.imageGrid}>
              <figure className={styles.figure}>
                <img
                  src="/hotformerloc/assets/dataset/experiments_benchmarking.png"
                  alt="Benchmarking Results"
                  className={styles.image}
                />
                <figcaption>
                  Recall@N curves of four SOTA LPR methods on CS-Wild-Places
                  Baseline and Unseen splits
                </figcaption>
              </figure>
            </div>
            <div className={styles.section}>
              <p className={styles.paragraph}>
                As per the figure above, we demonstrate the performance of the
                proposed HOTFormerLoc on our In-house dataset, trained for 100
                epochs with a LR of 8e^-4, reduced by a factor of 10 after 50
                epochs. On the baseline and unseen evaluation sets, HOTFormerLoc
                achieves an improvement in AR@1 of 5.5% - 11.5%, and an
                improvement in AR@1% of 3.6% - 4.5%, respectively.
              </p>
            </div>
          </div>
          <div className={styles.imageGrid}>
            <figure className={styles.figure}>
              <img
                src="/hotformerloc/assets/dataset/dataset_sota_comparison_1.png"
                alt="SOTA on CS-Campus3D Comparison"
                className={styles.image}
              />
              <figcaption>
                Comparison of SOTA on CS-Campus3D with groundonly queries, and
                ground + aerial database.
              </figcaption>
            </figure>
          </div>
          <div className={styles.section}>
            <p className={styles.paragraph}>
              As per the figure above, we present the evaluation results 535 on
              CS-Campus3D, training our method for 300 epochs with a LR of
              5e^-4, reduced by a factor of 10 after 250 epochs. Our approach
              shows an improvement of 6.8% and 5.7% in AR@1 and AR@1%,
              respectively.
            </p>
          </div>
          <div className={styles.imageGrid}>
            <figure className={styles.figure}>
              <img
                src="/hotformerloc/assets/dataset/dataset_wildplaces_comparison_1.png"
                alt="Comparison on Wild-Places"
                className={styles.image}
              />
              <figcaption>
                Comparison on Wild-Places. HOTFormerLoc† denotes cylindrical
                octree windows. LoGG3D-Net1 indicates training the network using
                a 256-dimensional global descriptor, as opposed to the
                1024-dimensional descriptor reported in Wild-Places
              </figcaption>
            </figure>
            <p className={styles.paragraph}>
              In the figure above, we report evaluation results on Wild-Places
              under the inter-sequence evaluation setting, training our method
              for 100 epochs with a LR of 3e^-3, reduced by a factor of 10 after
              30 epochs. LoGG3D-Net remains the highest performing method by a
              margin of 2.5% and 1.8% in AR@1 and MRR, respectively, but we
              achieve a gain of 5.5% and 3.5% in AR@1 and MRR over MinkLoc3Dv2.
              However, we note that LoGG3D-Net is trained on Wild-Places with a
              global descriptor size of 1024, compared to our compact descriptor
              of size 256.
            </p>
          </div>

          <div className={styles.imageGrid}>
            <figure className={styles.figure}>
              <img
                src="/hotformerloc/assets/dataset/dataset_sota_comparison_2.png"
                alt="Comparison of SOTA on Oxford RobotCar"
                className={styles.image}
              />
              <figcaption>
                Comparison of SOTA on Oxford RobotCar using the baseline
                evaluation setting and dataset
              </figcaption>
            </figure>
            <p className={styles.paragraph}>
              The table above reports evaluation results on Oxford Robot Car
              using the baseline evaluation setting and dataset intro duced by ,
              training our method for 150 epochs with a LR of 5e^-4, reduced by
              a factor of 10 after 100 epochs. We outperform previous SOTA
              methods, showing improved generalisation on the unseen R.A. and
              B.D. environments with an increase of 2.7% and 4.1% in AR@1,
              respectively.
            </p>
          </div>

          <div>
            <h3 id="ablation-study" className={styles.subHeading}>
              Ablation Study
            </h3>
            <div className={styles.imageGrid}>
              <figure className={styles.figure}>
                <img
                  src="/hotformerloc/assets/dataset/dataset_ablation_study_1.png"
                  alt="Ablation Study"
                  className={styles.image}
                />
                <figcaption>
                  Ablation study on the effectiveness of HOTFormer- Loc
                  components on Oxford, CS-Campus3D and In-house.
                </figcaption>
              </figure>
              <p className={styles.paragraph}>
                We provide ablations to verify the effectiveness of various
                HOTFormer- Loc components on Oxford, CS-Campus3D, and In-house.
                Disabling relay tokens results in a 2.5%-4.7% drop in
                performance across all datasets, highlighting the importance of
                global feature interactions within HOTFormerLoc.
              </p>
            </div>
          </div>
          <section className={styles.futureWork}>
            <h3 id="future-work" className={styles.sectionHeading}>
              Future Work
            </h3>
            <p className={styles.paragraph}>
              We propose HOTFormerLoc, a novel 3D place recognition method that
              leverages octree-based transformers to capture multi-granular
              features through both local and global in teractions. We introduce
              and discuss a new cross-source LPR benchmark, In-house, designed
              to advance research on re-localisation in challenging settings.
              Our method demonstrates superior performance on our In-house
              dataset and outperforms existing SOTA on LPR benchmarks for both
              ground and aerial views. Despite these advancements, cross-source
              LPR remains a promising area for further im provement on our
              In-house dataset. There remain avenues to improve HOTFormerLoc,
              such as token pruning to reduce redundant computations and
              enhancing feature learning with image data.
            </p>
          </section>
        </section>
      </main>
    </div>
  );
};

export default Paper;
