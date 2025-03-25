import React from "react";
import downloadStyles from "./Download.module.css";
import styles from "../../Common.module.css";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const Header: React.FC = () => (
  <header className={styles.header}>
    <h1 className={styles.title}>Downloads</h1>
  </header>
);

const CheckpointRow: React.FC<CheckpointRowProps> = ({ model_name, dataset, weights_name, link }) => (
  <tr>
    <td>{model_name}</td>
    <td>{dataset}</td>
    <td>
      <a
        href={link}
        className={downloadStyles.downloadLink}
        target="_blank"
        rel="noopener noreferrer"
      >
        {weights_name}
      </a>
    </td>
  </tr>
);

const Checkpoint: React.FC = () => {
  const checkpoints: CheckpointRowProps[] = [
    { model_name: "HOTFormerLoc", dataset: "CS-Wild-Places", weights_name: "hotformerloc_cs-wild-places.pth", link: "https://www.dropbox.com/scl/fi/bcgcmbyic591f3bviib64/hotformerloc_cs-wild-places.pth?rlkey=vrw0seq6nfbsihijbhqatll2u&st=d7enawjw&dl=0" },
    { model_name: "HOTFormerLoc", dataset: "CS-Campus3D", weights_name: "hotformerloc_cs-campus3D.pth", link: "https://www.dropbox.com/scl/fi/l9jyn5310gjf80zw35v7z/hotformerloc_cs-campus3d.pth?rlkey=s0bpcysyc1xt2357shhclpnlw&st=zhh679b9&dl=0" },
    { model_name: "HOTFormerLoc", dataset: "Wild-Places", weights_name: "hotformerloc_wild-places.pth", link: "https://www.dropbox.com/scl/fi/yd94iy9dq6k1m312ifnyx/hotformerloc_wild-places.pth?rlkey=5ndv0p48c7hyjvah90eab1l1e&st=zl1716hh&dl=0" },
    { model_name: "HOTFormerLoc", dataset: "Oxford RobotCar", weights_name: "hotformerloc_oxford.pth", link: "https://www.dropbox.com/scl/fi/4r3470zo9zomkyjys5nrm/hotformerloc_oxford.pth?rlkey=eocfo3yvmhuqqgsmjtypgf78s&st=ybhzcj6y&dl=0" },
    { model_name: "MinkLoc3Dv2", dataset: "CS-Wild-Places", weights_name: "minkloc3dv2_cs-wild-places.pth", link: "https://www.dropbox.com/scl/fi/2w4l8gv7qbmp0lh4eztsf/minkloc3dv2_cs-wild-places.pth?rlkey=udxvtkr6yfgdnyizra4gmw0qa&st=p0evrh61&dl=0" },
    { model_name: "CrossLoc3D", dataset: "CS-Wild-Places", weights_name: "crossloc3d_cs-wild-places.pth", link: "https://www.dropbox.com/scl/fi/5ikt1jvr2fabiaw8mhqbb/crossloc3d_cs-wild-places.pth?rlkey=lb4gp2n814im3twy4zy5d67bd&st=znup5ewi&dl=0" },
    { model_name: "LoGG3D-Net", dataset: "CS-Wild-Places", weights_name: "logg3dnet_cs-wild-places.pth", link: "https://www.dropbox.com/scl/fi/51se5akdyg35xy2dsrosj/logg3dnet_cs-wild-places.pth?rlkey=4nvvp8gw656wdbj3081jzcn0i&st=n5ytpnzc&dl=0" },
  ];

  return (
    <section className={styles.section}>
      <h2 id="checkpoint" className={styles.sectionHeading}>
        Checkpoints
      </h2>
      <p className={styles.description}>
        {/* The links in the table below will allow you to download checkpoints for
        our trained models on HOTFormerLoc, CrossLoc3D, LoGG3D-Net, and
        MinkLoc3Dv2 architectures, as described in the paper associated with
        this dataset release. */}
        Pre-trained weights for HOTFormerLoc and other models reported in the paper
        can be downloaded from the links below. You can download them individually
        below, or download and extract all from {" "}
        <a
          href="https://www.dropbox.com/scl/fi/qjyh966styqlye38a4c37/pretrained_weights.tar.gz?rlkey=qkuhupf3og7mfkfid8dts7xej&st=wx8q2v68&dl=0"
          target="_blank"
          rel="noopener noreferrer"
        >
          this link 
        </a>.
      </p>

      <table className={downloadStyles.downloadTable}>
        <thead>
          <tr>
            <th>Model</th>
            <th>Dataset</th>
            <th>Download</th>
          </tr>
        </thead>
        <tbody>
          {checkpoints.map((checkpoint, index) => (
            <CheckpointRow key={index} {...checkpoint} />
          ))}
        </tbody>
      </table>
    </section>
  );
};

const Dataset: React.FC = () => (
  <section className={styles.section}>
    <h2 id="dataset" className={styles.sectionHeading}>
      Dataset
    </h2>
    <p className={downloadStyles.datasetDescription}>
      Our dataset can be downloaded through the{" "}
      <a
        href="https://data.csiro.au/collection/csiro:64896"
        target="_blank"
        rel="noopener noreferrer"
      >
        CSIRO Data Access Portal
      </a>.
    </p>
  </section>
);

const UsageExamples: React.FC = () => {
  const sampleCode = `# Sample Python Code for Loading Submaps
import numpy as np
import open3d as o3d

# Load submap
point_cloud = o3d.io.read_point_cloud(submap_path)
pts = np.asarray(point_cloud.points)  # (N, 3)

# Visualise submap
o3d.visualization.draw_geometries([point_cloud])
`;

  return (
    <section className={styles.section}>
      <h2 id="usage-examples" className={styles.sectionHeading}>
        Usage Examples
      </h2>
      <SyntaxHighlighter language="python" style={materialDark} showLineNumbers>
        {sampleCode}
      </SyntaxHighlighter>
    </section>
  );
};

const Download: React.FC = () => (
  <div className={styles.container}>
    <Header />
    <main className={styles.main}>
      <Checkpoint />
      <Dataset />
      <UsageExamples />
    </main>
  </div>
);

export default Download;
