import React from "react";
import styles from "../../Common.module.css";
import homeStyles from "./Home.module.css";
import PCMVContainer from "../../components/PCMVContainer/PCMVContainer";
import ContentBlock from "../../components/ContentBlock/ContentBlock";
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
        ground-to-aerial scenarios across urban and forest environments. We propose:
      </p>
      <div className={homeStyles.textContainer}>
        <ul className={homeStyles.dotPoints}>
          <li>
            A novel octree-based <b>hierarchical attention</b> that efficiently relays long-range contextual information across multiple scales
          </li>
          <li>
            <b>Cylindrical Octree Attention</b> to better represent the variable density of spinning lidar point clouds
          </li>
          <li>
            <b>Pyramid Attention Pooling</b> to adaptively select and aggregate multi-scale local descriptors into a global descriptor for end-to-end LPR
          </li>
        </ul>
      </div>
      <ContentBlock
        imageSrc="/hotformerloc/assets/architecture/architecture_hotformerloc.png"
        altText="HOTFormerLoc Architecture"
        caption="HOTFormerLoc Architecture"
        description=""
      />
      <p className={styles.paragraph}>
        We also propose <b>CS-Wild-Places</b>, a novel dataset for ground-to-aerial
        lidar place recognition featuring point cloud data from ground and aerial
        lidar scans captured in dense forests. Our dataset features:
      </p>
      <div className={homeStyles.contentContainer}>
        <div className={homeStyles.textContainer}>
          <ul className={homeStyles.dotPoints}>
            <li>
              ~<b>100K</b> high resolution lidar submaps captured in <b>4 unique forests</b> over
              3 years
            </li>
            <li>
              Challenging representational gaps such as variable point density
              and significant occlusions between viewpoints
            </li>
            <li>
              Accurate <b>6-DoF</b> ground truth for all sequences
            </li>
          </ul>
        </div>
        <div className={homeStyles.videoContainer}>
          <VideoPlayer src="/hotformerloc/assets/visualisation/karawatha_submaps_matched_labelled.mp4" />
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
        Use CS-Wild-Places!
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
      <h2 id="citation" className={styles.sectionHeading}>
        Citation
      </h2>
      <p className={styles.paragraph}>If you find this work useful, consider citing our paper!</p>
      <SyntaxHighlighter language="LaTeX" style={materialDark}>
        {citationBibtex}
      </SyntaxHighlighter>
    </section>
  );
};

const Contact: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="contact" className={styles.sectionHeading}>
        Contact Us
      </h2>
      <p className={styles.paragraph}>
        If you have any feedback about the paper or dataset please contact <a href="ethan.griffiths@data61.csiro.au">ethan.griffiths@data61.csiro.au</a>.
      </p>
    </section>
  );
};

const Updates: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="updates" className={styles.sectionHeading}>
        Updates
      </h2>
      <p className={styles.paragraph}>
        <b>2025 Feb:</b> CS-Wild-Places dataset released.
      </p>
    </section>
  );
};

const Acknowledgements: React.FC = () => {
  return (
    <section className={styles.section}>
      <h2 id="acknowledgements" className={styles.sectionHeading}>
        Acknowledgements
      </h2>
      <p className={styles.paragraph}>
        We would like to acknowledge the authors of the Wild-Places dataset for
        their work, which serves as the foundation for CS-Wild-Places.
        We wish to acknowledge the support of the Research Engineering Facility
        (REF) team at QUT for the provision of expertise and research
        infrastructure in enablement of this project. We thank Hexagon for
        providing access to SmartNet RTK corrections service to support precise
        survey of GCPs. We further acknowledge the support of the Terrestrial
        Ecosystem Research Network (TERN), supported by the National Collaborative
        Infrastructure Strategy (NCRIS). Additional funding was provided
        through the CSIRO's Digital Water and Landscapes initiative (3D-AGB project).
      </p>
    </section>
  );
};

const Home = () => {
  return (
    <div className={styles.container}>
      <Header />
      <main className={styles.main}>
        <Links />
        <GithubLink />
        <Overview />
        <Visualise />
        <Citation />
        <Contact />
        <Updates />
        <Acknowledgements />
      </main>
    </div>
  );
};

export default Home;
