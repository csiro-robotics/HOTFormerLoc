import React from "react";
import styles from "../../Common.module.css";
import homeStyles from "./Home.module.css";
import PCMVContainer from "../../components/PCMVContainer/PCMVContainer";
import { useNavigate } from "react-router-dom";
import VideoPlayer from "../../components/VideoPlayer/VideoPlayer";
const Header: React.FC = () => {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>
        Hierarchical Octree Transformer for Lidar Place Recognition
      </h1>
    </header>
  );
};

const Overview: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="overview" className={styles.sectionHeading}>
        Overview
      </h2>
      <p className={styles.paragraph}>
        We present HOTFormerLoc, a novel and versatile Hierarchical Octree-based
        Transformer for large-scale 3D place recognition in both
        ground-to-ground and ground-to-aerial scenarios across urban and forest
        environments.
      </p>
      <div className={homeStyles.contentContainer}>
        <div className={homeStyles.textContainer}>
          <ul className={homeStyles.dotPoints}>
            <li>
              HOTFormerLoc: A novel Hierarchical Octree-based Transformer for
              large-scale 3D place recognition.
            </li>
            <li>The datset: XYZ</li>
            <li>The Download page:</li>
          </ul>
        </div>
        <div className={homeStyles.videoContainer}>
          <VideoPlayer src="/hotformerloc/assets/visualisation/karawatha_aerial_vid.mp4" />
        </div>
      </div>
    </section>
  );
};

const Visualise: React.FC = () => {
  return (
    <div>
      <section className={styles.section}>
        <h2 id="visualise-submap" className={styles.sectionHeading}>
          Visualise a Submap
        </h2>
        <PCMVContainer
          title1="Aerial View"
          title2="Ground View"
          isSingleViewer={false}
        />
      </section>
    </div>
  );
};

const Links: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className={styles.buttonContainer}>
      <button className={styles.navButton} onClick={() => navigate("/paper")}>
        Checkout the Paper!
      </button>

      <button className={styles.navButton} onClick={() => navigate("/dataset")}>
        View the Dataset!
      </button>
    </div>
  );
};
const Citation: React.FC = () => {
  return (
    <section className={styles.futureWork}>
      <h3 id="citation" className={styles.sectionHeading}>
        Citation
      </h3>
      <p className={styles.paragraph}>The people who wrote the paper here!</p>
    </section>
  );
};
const Home = () => {
  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.main}>
        <Overview />
        <Links />
        <Visualise />
        <Citation/>
      </main>
    </div>
  );
};

export default Home;
