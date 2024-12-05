import styles from "./Acknowledgements.module.css";

const Acknowledgements = () => {
  return (
    <>
      <div className={styles.container}>
        <header className={styles.header}>
          <h1 className={styles.title}>Acknowledgements</h1>
        </header>
        <main className={styles.main}>
          <section className={styles.section}>
            <p className={styles.paragraph}>
              We wish to acknowledge the support of the Research Engineering
              Facility (REF) team at QUT for the provision of expertise and
              research infrastructure in enablement of this project. We thank
              Hexagon for providing access to SmartNet RTK corrections service
              to support precise survey of GCPs. We further acknowledge the
              support of the Terrestrial Ecosystem Research Network (TERN),
              supported by the National Collaborative Infrastructure Strategy
              (NCRIS). (Researcher initials MR, PM, SL, SJ etc) received funding
              through the CSIRO's Digital Water and Landscapes initiative
              (3D-AGB project).
            </p>
          </section>
        </main>
      </div>
    </>
  );
};

export default Acknowledgements;