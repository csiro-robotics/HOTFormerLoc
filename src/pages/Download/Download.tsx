import React from "react";
import downloadStyles from "./Download.module.css";
import styles from "../../Common.module.css";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { materialDark } from "react-syntax-highlighter/dist/esm/styles/prism";

const Header: React.FC = () => (
  <header className={styles.header}>
    <h1 className={styles.title}>Download</h1>
  </header>
);

const CheckpointRow: React.FC<CheckpointRowProps> = ({ name, link }) => (
  <tr>
    <td>{name}</td>
    <td>
      <a
        href={link}
        className={downloadStyles.downloadLink}
        target="_blank"
        rel="noopener noreferrer"
      >
        Download
      </a>
    </td>
  </tr>
);

const Checkpoint: React.FC = () => {
  const checkpoints: CheckpointRowProps[] = [
    { name: "HOTFormerLoc", link: "" },
    { name: "CrossLoc3D", link: "" },
    { name: "LoGG3D-Net", link: "" },
    { name: "MinkLoc3Dv2", link: "" },
  ];

  return (
    <section className={styles.section}>
      <h2 id="checkpoint" className={styles.sectionHeading}>
        Checkpoint
      </h2>
      <p className={styles.description}>
        The links in the table below will allow you to download checkpoints for
        our trained models on HOTFormerLoc, CrossLoc3D, LoGG3D-Net, and
        MinkLoc3Dv2 architectures, as described in the paper associated with
        this dataset release.
      </p>

      <table className={downloadStyles.downloadTable}>
        <thead>
          <tr>
            <th>Name</th>
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
        href="https://data.csiro.au/"
        target="_blank"
        rel="noopener noreferrer"
      >
        CSIRO Data Access Portal
      </a>
      .
    </p>
  </section>
);

const UsageExamples: React.FC = () => {
  const sampleCode = `# Sample Python Code for Loading Data
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = np.load("sample_data.npy")

# Visualise
plt.imshow(data, cmap="viridis")
plt.title("Sample Dataset Visualisation")
plt.show()
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
