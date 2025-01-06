import React from "react";
import styles from "../../Common.module.css";
import homeStyles from "./Home.module.css";
import PCMVContainer from "../../components/PCMVContainer/PCMVContainer";
import { useNavigate } from "react-router-dom";
import VideoPlayer from "../../components/VideoPlayer/VideoPlayer";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/esm/styles/prism";
const Header: React.FC = () => {
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>
        HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place
        Recognition Across Ground and Aerial Views
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
        We present <b>HOTFormerLoc</b>, a novel and versatile <b>H</b>ierarchical <b>O</b>ctree-based <b>T</b>rans<b>former</b> for
        large-scale lidar place recognition in both ground-to-ground and
        ground-to-aerial scenarios across urban and forest environments.
      </p>
      <p className={styles.paragraph}>
        We also propose <b>CS-Wild-Places</b>, a novel dataset for ground-to-aerial
        lidar place recognition featuring point cloud data from ground and aerial
        lidar scans captured in dense forests. Our dataset features:
      </p>
      <div className={homeStyles.contentContainer}>
        <div className={homeStyles.textContainer}>
          <ul className={homeStyles.dotPoints}>
            <li>
              ~100K high resolution lidar submaps captured in 4 unique forests over
              3 years
            </li>
            <li>
              Challenging representational gaps such as variable point density
              and significant occlusions between viewpoints
            </li>
            <li>
              Accurate 6-DoF ground truth for all sequences
            </li>
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
        Interactive Submap Visualisation
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

const GithubLink: React.FC = () => {
  return (
    <div className={styles.buttonContainer}>
      <a
        href="https://github.com/csiro-robotics/HOTFormerLoc"
        target="_blank"
        rel="noopener noreferrer"
        className={styles.navButton}
      >
        Visit our GitHub Repo!
      </a>
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
  const citationBibtex = `@InProceedings{HOTFormerLoc,
	author    = {Griffiths, Ethan and Haghighat, Maryam and Denman, Simon and Fookes, Clinton and Ramezani, Milad},
	title     = {HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place Recognition Across Ground and Aerial Views},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month     = {todo},
	year      = {2025},
	pages     = {todo}
}`;
  return (
    <section className={styles.futureWork}>
      <h3 id="citation" className={styles.sectionHeading}>
        Citation
      </h3>
      <p className={styles.paragraph}>If you find this work useful, consider citing our paper!</p>
      <SyntaxHighlighter language="LaTeX" style={materialDark}>
        {citationBibtex}
      </SyntaxHighlighter>
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
        <GithubLink />
        <Visualise />
        <Citation/>
      </main>
    </div>
  );
};

export default Home;
