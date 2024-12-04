import React from "react";
import Slideshow from "../../components/Slideshow/Slideshow";
import styles from "./Paper.module.css";

const images = [
  "/hotformerloc/assets/karawatha_image_1.png",
  "/hotformerloc/assets/karawatha_image_2.png",
  "/hotformerloc/assets/karawatha_image_3.png",
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
                src="/hotformerloc/assets/architecture_hotformerloc.png"
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
                  src="/hotformerloc/assets/architecture_rtsa.png"
                  alt="RTSA Block Architecture Diagram"
                  className={styles.image}
                />
                <figcaption>RTSA Block Diagram</figcaption>
              </figure>
              <figure className={styles.figure}>
                <img
                  src="/hotformerloc/assets/architecture_rtsa_2.png"
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
                  src="/hotformerloc/assets/architecture_hosa.png"
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
                src="/hotformerloc/assets/architecture_coa_2.png"
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
              Some text about Dataset Evaluation Criteria!
            </p>
          </div>

          <div>
            <h3 id="ablation-study" className={styles.subHeading}>
              Ablation Study
            </h3>
            <p className={styles.paragraph}>Some text about Ablation Study</p>
          </div>

          <section className={styles.futureWork}>
            <h3 id="future-work" className={styles.sectionHeading}>
              Future Work
            </h3>
            <p className={styles.paragraph}>Some text about Future Work</p>
          </section>
        </section>
      </main>
    </div>
  );
};

export default Paper;
