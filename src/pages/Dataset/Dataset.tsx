import React from "react";
import styles from "./Dataset.module.css"
const Dataset: React.FC = () => {
  return (
    <>
     <div className={styles.container}>
      <header className={styles.header}>
        <h1 className={styles.title}>
          Dataset
        </h1>
      </header>
      <main className={styles.main}>

        <section className={styles.section}>
          <h2 id="overview" className={styles.sectionHeading}>
            Overview
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
          <h2 id="methodology" className={styles.sectionHeading}>
            Data Collection Methodology
          </h2>
          <p className={styles.paragraph}>
          </p>
        </section>

        <section className={styles.section}>
          <h2 id="benchmark" className={styles.sectionHeading}>
            Benchmarking
          </h2>
          <p className={styles.paragraph}>
          </p>
        </section>

        <section className={styles.section}>
          <h2 id="model-images" className={styles.sectionHeading}>
            Model Images
          </h2>
          <p className={styles.paragraph}>
          </p>
        </section>

        </main>
    </div>
    </>
  );
};

export default Dataset;
