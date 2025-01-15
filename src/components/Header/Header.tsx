import React from "react";
import styles from "../../Common.module.css";
import AuthorsBar from "../AuthorsBar/AuthorsBar";

const Header: React.FC = () => {
  const authors = [
    {
      firstName: "Ethan",
      lastName: "Griffiths",
      university: "QUT, CSIRO Data61",
      link: "https://scholar.google.com/citations?user=a6BiSqoAAAAJ&hl=en",
    },
    {
      firstName: "Maryam",
      lastName: "Haghighat",
      university: "QUT",
      link: "https://scholar.google.com/citations?user=oJDmGg4AAAAJ&hl=en",
    },
    {
      firstName: "Simon",
      lastName: "Denman",
      university: "QUT",
      link: "https://scholar.google.com/citations?user=2bkcCykAAAAJ&hl=en",
    },
    {
      firstName: "Clinton",
      lastName: "Fookes",
      university: "QUT",
      link: "https://scholar.google.com.au/citations?user=VpaJsNQAAAAJ&hl=en",
    },
    {
      firstName: "Milad",
      lastName: "Ramezani",
      university: "CSIRO Data61",
      link: "https://scholar.google.com/citations?user=fn-lMpMAAAAJ&hl=en",
    },
  ];
  return (
    <header className={styles.header}>
      <h1 className={styles.title}>
        HOTFormerLoc: Hierarchical Octree Transformer for Versatile Lidar Place
        Recognition Across Ground and Aerial Views
      </h1>
      <h4 className={styles.subtitle}>
      IEEE/CVF Conference on Computer Vision and Pattern Recognition 2025 (CVPR 2025)
      </h4>
      <AuthorsBar authors={authors} />
    </header>
  );
};
export default Header;
