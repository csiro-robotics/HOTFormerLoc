import styles from "../../Common.module.css"

const Citation = () => {
  return (
    <>
      <div className={styles.container}>
        <header className={styles.header}>
          <h1 className={styles.title}>Citation</h1>
        </header>
        <main className={styles.main}>
          <section className={styles.section}>
            <p className={styles.paragraph}>
              People from the paper!
            </p>
          </section>
        </main>
      </div>
    </>
  );
};

export default Citation;
