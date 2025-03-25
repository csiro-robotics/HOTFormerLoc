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
      link: "https://www.qut.edu.au/about/our-people/academic-profiles/maryam.haghighat",
    },
    {
      firstName: "Simon",
      lastName: "Denman",
      university: "QUT",
      link: "https://www.qut.edu.au/about/our-people/academic-profiles/s.denman",
    },
    {
      firstName: "Clinton",
      lastName: "Fookes",
      university: "QUT",
      link: "https://www.qut.edu.au/about/our-people/academic-profiles/c.fookes",
    },
    {
      firstName: "Milad",
      lastName: "Ramezani",
      university: "CSIRO Data61",
      link: "https://people.csiro.au/r/m/milad-ramezani",
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
