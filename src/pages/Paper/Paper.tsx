import React from "react";
import ContentBlock from "../../components/ContentBlock/ContentBlock";
import styles from "../../Common.module.css";
import ImageWithCaption from "../../components/ImageWithCaption/ImageWithCaption";
import Header from "../../components/Header/Header";
import { useNavigate } from "react-router-dom";

const Summary: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="summary" className={styles.sectionHeading}>
        Summary
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
        We use an octree structure to represent point cloud submaps, which
        provides an inherent spatial hierarchy. We use this structure to
        generate a hierarchical feature pyramid, which is tokenised and
        partitioned into local attention windows of size k inspired by the 
        octree-based attention introduced in{" "}
        <a
          href="https://dl.acm.org/doi/abs/10.1145/3592131"
          target="_blank"
          rel="noopener noreferrer"
        >
          OctFormer (Wang, 2023)
        </a>
        . To efficiently capture long range interactions that are missed by
        window attention, we introduce a set of <b>relay tokens</b> which
        summarise local regions at multiple scales. Both local and relay tokens
        are processed in a series of HOTFormer blocks, iteratively refining
        features in a global-to-local fashion. A pyramid attention pooling layer
        then aggregates the multi-scale features into a single global
        descriptor.
      </p>

      <div className={styles.imageGrid}>
        <h3 id="hotformerloc" className={styles.subHeading}>
          HOTFormerLoc
        </h3>
        <ImageWithCaption
          src="/hotformerloc/assets/architecture/architecture_hotformerloc.png"
          alt="HOTFormerLoc Architecture"
          caption="HOTFormerLoc Architecture"
        />
      </div>

      <div>
        <h3 id="hotformerblock" className={styles.subHeading}>
          RTSA and H-OSA Layers
        </h3>
        <p className={styles.paragraph}>
          HOTFormer blocks consist of relay token self-attention (RTSA) to
          induce long-distance multi-scale interactions, and hierarchical octree
          self-attention (H-OSA) to refine local features and propagate global
          contextual cues learned by the relay tokens.
        </p>
        <ImageWithCaption
          src="/hotformerloc/assets/architecture/architecture_rtsa_hosa.png"
          alt="HOTFormer Block Components"
          caption="HOTFormer Block Components"
          width="60%"
        // description=""
        />
        <p className={styles.paragraph}>
          We compare the attention patterns of a query token (red) from the
          first, middle, and last RTSA layer in the network (yellow indicates
          high attention, purple indicates weak attention). We see that RTSA
          learns a local-to-global attention pattern as it progresses through
          the network, focusing on fine-grained neighbouring relay tokens in
          early layers, and progressively widening and coarsening receptive
          field in later layers.
        </p>
        <ImageWithCaption
          src="/hotformerloc/assets/architecture/architecture_rtsa_2.png"
          alt="RTSA Attention Visualisation"
          caption="Relay token multi-scale attention visualised on the octree feature pyramid"
          width="80%"
        />
      </div>

      <div>
        <h3 id="coa" className={styles.subHeading}>
          Cylindrical Octree Attention
        </h3>
        <p className={styles.paragraph}>
          {/* Cartesian VS cylindrical attention window serialisation (each window
          indicated by the arrow colour) for the 2D equivalent of an octree with
          depth d = 3 and window size k = 7. Cylindrical octree attention
          windows better represent the distribution of spinning lidar point
          clouds. */}
          Given that raw lidar scans suffer from increased sparsity with
          distance from the sensor, we propose a simple yet effective
          modification to the octree structure, particularly suited to the
          circular pattern of spinning lidar. Typically, octrees are constructed
          in Cartesian coordinates, where the (x, y, z) dimensions are
          subdivided to form octants. Instead, we construct octrees in
          cylindrical coordinates, i.e. (&rho;, &theta;, z), to better reflect
          the distribution of lidar point clouds captured from the ground.
        </p>
        <p className={styles.paragraph}>
          The effect of this on the 2D-equivalent of an octree is seen below.
          Cartesian octree attention windows cover a uniform area and ignore the
          underlying point density, whereas the cylindrical octree attention
          windows respect the point distribution, with fine-grained windows near
          the sensor, and sparser windows at lower-density regions further away.
        </p>
        <ImageWithCaption
          src="/hotformerloc/assets/architecture/architecture_coa_2.png"
          alt="Cylindrical Octree Attention Architecture Diagram"
          caption="Cylindrical Octree Attention Comparison."
          width="75%"
        />
      </div>

      <div>
        <h3 id="pap" className={styles.subHeading}>
          Pyramid Attention Pooling
        </h3>
        <p className={styles.paragraph}>
          To best utilise the set of hierarchical octree features, we propose a
          novel pyramid attentional pooling mechanism to aggregate
          multi-resolution tokens into a distinctive global descriptor whilst
          adaptively filtering out irrelevant tokens. We introduce a set of
          learnable query tokens at each resolution to provide flexibility for
          the network to identify distinctive clusters of tokens that best
          represent the environment. Importantly, this approach can handle the
          variable number of local tokens from each level whilst retaining
          linear computational complexity.
        </p>
      </div>
    </section>
  );
};

const Experiments: React.FC = () => {
  return (
    <section className={styles.section} id="experiments">
      <h2 className={styles.sectionHeading}>Experiments</h2>
      <p className={styles.paragraph}>
        We conduct experiments on a suite of LPR benchmarks, to demonstrate the
        versatility of HOTFormerLoc to different environments and scenarios. We
        also provide ablation studies that verify the effectiveness of
        HOTFormerLoc, which can be found in the main paper.
      </p>

      <div>
        <h3 id="sota-comparison" className={styles.subHeading}>
          Comparison to SOTA
        </h3>
        <p className={styles.paragraph}>
          To demonstrate our method's versatility, we conduct experiments on our
          novel CS-Wild-Places dataset, as well as CS-Campus3D, Wild-Places, and
          Oxford RobotCar, using the established training and testing splits for
          each.
        </p>

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/experiments_benchmarking.png"
          altText="CS-Wild-Places Benchmarking Results"
          caption="Recall@N curves of SOTA LPR methods on CS-Wild-Places Baseline and Unseen splits."
          description="We demonstrate the performance of the proposed HOTFormerLoc on our CS-Wild-Places dataset.
           On the baseline and unseen evaluation sets, HOTFormerLoc achieves an improvement in AR@1 of 5.5% and 11.5%,
            and an improvement in AR@1% of 3.6% and 4.5%, respectively."
          width="90%"
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_sota_comparison_1.png"
          altText="SOTA on CS-Campus3D Comparison"
          caption="Comparison of SOTA on CS-Campus3D with ground-only queries and aerial-only database."
          description="We present the evaluation results on CS-Campus3D. 
           Our approach shows an improvement of 9.7% and 9.2% in AR@1 and AR@1%, 
            respectively, outperforming CrossLoc3D which is specifically designed 
             for the ground-aerial LPR problem."
          width="70%"
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_wildplaces_comparison_1.png"
          altText="Comparison on Wild-Places."
          caption="Comparison of SOTA on Wild-Places, where LoGG3D-Net^1 is trained using a 256-dimensional global descriptor for fair comparison,
           as opposed to the 1024-dimensional descriptor reported in Wild-Places."
          description="We report evaluation results on Wild-Places under the inter-sequence evaluation setting.
            LoGG3D-Net remains the highest performing method by a margin of 2.5% and 1.8% in AR@1 and MRR, respectively,
            but we achieve a gain of 5.5% and 3.5% in AR@1 and MRR over MinkLoc3Dv2. We note that LoGG3D-Net is
            trained on Wild-Places with a global descriptor size of 1024, compared to our compact descriptor of size 256."
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_sota_comparison_2.png"
          altText="Comparison of SOTA on Oxford RobotCar"
          caption="Comparison of SOTA on Oxford RobotCar using the baseline evaluation setting and dataset proposed in PointNetVLAD."
          description="We report evaluation results on Oxford RobotCar under the baseline evaluation setting. We outperform previous SOTA methods, showing improved generalisation on the unseen R.A. and B.D. environments with an increase of 2.7% and 4.1% in AR@1, respectively."
        />
      </div>

      {/* <div>
        <h3 id="ablation-study" className={styles.subHeading}>
          Ablation Study
        </h3>
        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_ablation_study_1.png"
          altText="Ablation Study"
          caption="Ablation study on the effectiveness of HOTFormerLoc components on Oxford, CS-Campus3D, and CS-Wild-Places."
          description="We provide ablations to verify the effectiveness of various HOTFormerLoc components on Oxford, CS-Campus3D, and CS-Wild-Places.
            Disabling relay tokens results in a 2.5%-4.7% drop in performance across all datasets, highlighting the importance of global feature interactions within HOTFormerLoc.
            The importance of our pyramid attention pooling is also clear, with a 2.7%-22.5% drop in performance when using alternative GeM-based pooling methods."
        />

        <ContentBlock
          imageSrc="/hotformerloc/assets/dataset/dataset_ablation_study_cyl.png"
          altText="Cylindrical Octree Ablation"
          caption="Ablation study considering cartesian vs cylindrical octree attention windows on Wild-Places."
          description="Cylindrical octree attention windows are essential for
            ground-captured lidar scans in natural environments, contributing to
            a significant improvement in AR@1 and MRR of 14.6% and 10.5% on
            Karawatha, and 13.8% and 9.4% on Venman, compared to Cartesian
            octree attention windows. "
        />
      </div> */}
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
        While our method demonstrates superior performance on our CS-Wild-Places
        dataset and outperforms existing SOTA on LPR benchmarks for both ground
        and aerial views, cross-source LPR remains a promising area for further
        improvement. There remain avenues to improve HOTFormerLoc, such as token
        pruning to reduce redundant computations and enhancing feature learning
        by introducing multi-modality through image data.
      </p>
    </section>
  );
};

const Links: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className={styles.buttonContainer}>
      <a
        href="https://arxiv.org/abs/2503.08140"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          textDecoration: 'none',
          color: '#FFF',
        }}
        className={styles.navButton}
      >
        arXiv
      </a>

      <a
        href="https://github.com/csiro-robotics/HOTFormerLoc"
        target="_blank"
        rel="noopener noreferrer"
        style={{
          textDecoration: 'none',
          color: '#FFF',
        }}
        className={styles.navButton}
      >
        GitHub
      </a>
      <button className={styles.navButton} onClick={() => navigate("/download")}>
        Checkpoint Download
      </button>
    </div>
  );
};

const Paper: React.FC = () => {
  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.main}>
        <Links />
        <Summary />
        <ImageWithCaption
          src="/hotformerloc/assets/dataset/dataset_model_comparison.png"
          alt="HOTFormerLoc performance comparison"
          caption="    HOTFormerLoc achieves SOTA performance across a suite of LPR
          benchmarks with diverse environments, varying viewpoints, and
          different point cloud densities"
          width="70%"
        />
        <NetworkArchitecture />
        <Experiments />
        <FutureWork />
      </main>
    </div>
  );
};

export default Paper;
