import React from "react";
import ImageWithCaption from "../../components/ImageWithCaption/ImageWithCaption";
import ContentBlock from "../../components/ContentBlock/ContentBlock";
import styles from "../../Common.module.css";

const Header: React.FC = () => {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>
        HOTFormerLoc: Hierarchical Octree Transformer for Lidar Place
        Recognition
      </h1>
    </header>
  );
};

const DownloadLink: React.FC = () => {
  return (
    <div className={styles.section}>
      <a>Download the full paper here!</a>
    </div>
  );
};

const Abstract: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="abstract" className={styles.sectionHeading}>
        Abstract
      </h2>
      <p className={styles.paragraph}>
        We present HOTFormerLoc, a novel and versatile Hierarchical Octree-based
        Transformer for large-scale 3D place recognition in both
        ground-to-ground and ground-to-aerial scenarios across urban and forest
        environments. Leveraging an octree-based structure, we propose a
        multi-scale attention mechanism that captures spatial and semantic
        features across granularities.
      </p>
    </section>
  );
};

const NetworkArchitecture: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="network-architecture" className={styles.sectionHeading}>
        Network Architecture
      </h2>
      <p className={styles.paragraph}>
        We use an octree to generate a hierarchical feature pyramid F, which is
        tokenised and partitioned into local attention windows F̂l of size k (k =
        3 in this example). We introduce a set of relay tokens RT_l representing
        local regions at each level and process both local and relay tokens in a
        series of HOTFormer blocks. A pyramid attention pooling layer then
        aggregates the multi-scale features into a single global descriptor.
      </p>

      <div className={styles.imageGrid}>
        <h3 id="hotformerloc" className={styles.subHeading}>
          HOTFormerLoc
        </h3>
        <ContentBlock
          imageSrc="/hotformerloc/assets/architecture/architecture_hotformerloc.png"
          altText="HOTFormerLoc Architecture"
          caption="HOTFormerLoc Architecture"
          description=""
        />
      </div>

      <div>
        <h3 id="rtsa" className={styles.subHeading}>
          Relay Token Self-Attention (RTSA) Block
        </h3>
        <p className={styles.paragraph}>
          HOTFormer blocks consist of relay token self-attention (RTSA) to
          induce long-distance multi-scale interactions.
        </p>
        <ContentBlock
          imageSrc="/hotformerloc/assets/architecture/architecture_rtsa.png"
          altText="RTSA Block Architecture Diagram"
          caption="RTSA Block Diagram"
          description=""
        />
        <ContentBlock
          imageSrc="/hotformerloc/assets/architecture/architecture_rtsa_2.png"
          altText="RTSA Attention Visualization"
          caption="Relay token multi-scale attention visualized on the octree feature pyramid."
          description=""
        />
      </div>

      <div>
        <h3 id="hosa" className={styles.subHeading}>
          Hierarchical Octree Self-Attention (HOSA) Block
        </h3>
        <p className={styles.paragraph}>
          HOTFormer blocks also consist of hierarchical octree self-attention
          (HOSA) to refine local features and propagate global contextual cues
          learned by the relay tokens.
        </p>
        <ContentBlock
          imageSrc="/hotformerloc/assets/architecture/architecture_hosa.png"
          altText="HOSA Block Architecture Diagram"
          caption="HOSA Block Diagram"
          description=""
        />
      </div>

      <div>
        <h3 id="coa" className={styles.subHeading}>
          Cylindrical Octree Attention (COA)
        </h3>
        <p className={styles.paragraph}>
          Cartesian VS cylindrical attention window serialisation (each window
          indicated by the arrow colour) for the 2D equivalent of an octree with
          depth d = 3 and window size k = 7. Cylindrical octree attention
          windows better represent the distribution of spinning lidar point
          clouds.
        </p>
        <ContentBlock
          imageSrc="/hotformerloc/assets/architecture/architecture_coa_2.png"
          altText="Cylindrical Octree Attention Architecture Diagram"
          caption="Cylindrical Octree Attention Diagram"
          description=""
        />
      </div>

      <div>
        <h3 id="pap" className={styles.subHeading}>
          Pyramid Attention Pooling (PAP)
        </h3>
        <p className={styles.paragraph}>Pyramid Attention Pooling</p>
      </div>
    </section>
  );
};

const Experiments: React.FC = () => {
  return (
    <section className={styles.section} id="experiments">
      <h2 className={styles.sectionHeading}>Experiments</h2>
      <p className={styles.paragraph}>
        This section explores the datasets and evaluation criteria used for our
        experiments, along with insights gained from ablation studies.
      </p>

      <div>
        <h3 id="evaluation-criteria" className={styles.subHeading}>
          Datasets and Evaluation Criteria
        </h3>
        <p className={styles.paragraph}>
          To demonstrate our method's versatility, we conduct experiments on
          Oxford RobotCar, CS-Campus3D, and Wild-Places, using the established
          training and testing splits for each.
        </p>

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_sota_comparison_1.png"
          altText="SOTA on CS-Campus3D Comparison"
          caption="Comparison of SOTA on CS-Campus3D with ground-only queries, and ground + aerial database."
          description="As per the figure above, we present the evaluation results on CS-Campus3D, training our method for 300 epochs with a LR of 5e^-4, reduced by a factor of 10 after 250 epochs. Our approach shows an improvement of 6.8% and 5.7% in AR@1 and AR@1%, respectively."
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/experiments_benchmarking.png"
          altText="Benchmarking Results"
          caption="Recall@N curves of four SOTA LPR methods on CS-Wild-Places Baseline and Unseen splits"
          description="As per the figure above, we demonstrate the performance of the proposed HOTFormerLoc on our In-house dataset, trained for 100 epochs with a LR of 8e^-4, reduced by a factor of 10 after 50 epochs. On the baseline and unseen evaluation sets, HOTFormerLoc achieves an improvement in AR@1 of 5.5% - 11.5%, and an improvement in AR@1% of 3.6% - 4.5%, respectively."
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_wildplaces_comparison_1.png"
          altText="Comparison on Wild-Places"
          caption="Comparison on Wild-Places. HOTFormerLoc† denotes cylindrical octree windows. LoGG3D-Net1 indicates training the network using a 256-dimensional global descriptor, as opposed to the 1024-dimensional descriptor reported in Wild-Places."
          description="In the figure above, we report evaluation results on Wild-Places under the inter-sequence evaluation setting, training our method for 100 epochs with a LR of 3e^-3, reduced by a factor of 10 after 30 epochs. LoGG3D-Net remains the highest performing method by a margin of 2.5% and 1.8% in AR@1 and MRR, respectively, but we achieve a gain of 5.5% and 3.5% in AR@1 and MRR over MinkLoc3Dv2."
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_sota_comparison_2.png"
          altText="Comparison of SOTA on Oxford RobotCar"
          caption="Comparison of SOTA on Oxford RobotCar using the baseline evaluation setting and dataset."
          description="The table above reports evaluation results on Oxford RobotCar using the baseline evaluation setting and dataset introduced by us, training our method for 150 epochs with a LR of 5e^-4, reduced by a factor of 10 after 100 epochs. We outperform previous SOTA methods, showing improved generalisation on the unseen R.A. and B.D. environments with an increase of 2.7% and 4.1% in AR@1, respectively."
        />
      </div>

      <div>
        <h3 id="ablation-study" className={styles.subHeading}>
          Ablation Study
        </h3>
        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_ablation_study_1.png"
          altText="Ablation Study"
          caption="Ablation study on the effectiveness of HOTFormerLoc components on Oxford, CS-Campus3D, and In-house."
          description="We provide ablations to verify the effectiveness of various HOTFormerLoc components on Oxford, CS-Campus3D, and In-house. Disabling relay tokens results in a 2.5%-4.7% drop in performance across all datasets, highlighting the importance of global feature interactions within HOTFormerLoc."
        />
      </div>
    </section>
  );
};

const FutureWork: React.FC = () => {
  return (
    <section className={styles.futureWork}>
      <h3 id="future-work" className={styles.sectionHeading}>
        Future Work
      </h3>
      <p className={styles.paragraph}>
        We propose HOTFormerLoc, a novel 3D place recognition method that
        leverages octree-based transformers to capture multi-granular features
        through both local and global in teractions. We introduce and discuss a
        new cross-source LPR benchmark, In-house, designed to advance research
        on re-localisation in challenging settings. Our method demonstrates
        superior performance on our In-house dataset and outperforms existing
        SOTA on LPR benchmarks for both ground and aerial views. Despite these
        advancements, cross-source LPR remains a promising area for further im
        provement on our In-house dataset. There remain avenues to improve
        HOTFormerLoc, such as token pruning to reduce redundant computations and
        enhancing feature learning with image data.
      </p>
    </section>
  );
};

const Paper: React.FC = () => {
  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.main}>
        <DownloadLink />
        <div className={styles.section}>
          <ImageWithCaption
            src="/hotformerloc/assets/dataset/dataset_model_comparison.svg"
            caption="HOTFormerLoc achieves SOTA performance across a suite of LPR
              benchmarks with diverse environments, varying viewpoints, and
              different point cloud densities."
            alt="Comparing Different Models"
          />
        </div>
        <Abstract />
        <NetworkArchitecture />
        <Experiments />
        <FutureWork />
      </main>
    </div>
  );
};

export default Paper;
