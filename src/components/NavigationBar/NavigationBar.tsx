import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Navbar, Nav, NavDropdown, Container } from "react-bootstrap";
import "bootstrap/dist/css/bootstrap.min.css";

const navigationItems: NavigationItem[] = [
  {
    label: "Home",
    path: "/",
    children: [
      { label: "Overview", path: "/#overview" },
      { label: "Visualise Submap", path: "/#visualise-submap" },
      { label: "Citation", path: "/#citation" },
    ],
  },
  {
    label: "Paper",
    path: "/paper",
    children: [
      { label: "Summary", path: "/paper#summary" },
      {
        label: "Network Architecture",
        path: "/paper#network-architecture",
        children: [
          { label: "HOTFormerLoc", path: "/paper#hotformerloc" },
          { label: "Cyndrical Octree Attention", path: "/paper#coa" },
          { label: "Pyramid Attention Pooling", path: "/paper#pap" },
        ],
      },
      {
        label: "Experiments",
        path: "/paper#experiments",
        children: [
          {
            label: "Comparison to SOTA",
            path: "/paper#sota-comparison",
          },
          { label: "Ablation Study", path: "/paper#ablation-study" },
        ],
      },
      { label: "Future Work", path: "/paper#future-work" },
    ],
  },
  {
    label: "Dataset",
    path: "/dataset",
    children: [
      { label: "Overview", path: "/dataset#overview" },
      { label: "Visualisation", path: "/dataset#visualisation" },
      { label: "Methodology", path: "/dataset#methodology" },
      { label: "Benchmarking", path: "/dataset#benchmarking" },
      { label: "Acknowledgements", path: "/dataset#acknowledgements" },
    ],
  },
  {
    label: "Download",
    path: "/download",
    children: [
      { label: "Checkpoint", path: "/download#checkpoint" },
      { label: "Dataset", path: "/download#dataset" },
      { label: "Usage Examples", path: "/download#usage-examples" },
    ],
  },
];

const NavBar: React.FC = () => {
  const navigate = useNavigate();
  const [openDropdown, setOpenDropdown] = useState<string | null>(null);

  const handleNavigation = (path: string) => {
    navigate(path);
    const hashIndex = path.indexOf("#");
    if (hashIndex !== -1) {
      const id = path.slice(hashIndex + 1);
      setTimeout(() => {
        const element = document.getElementById(id);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 100);
    }
  };

  return (
    <Navbar
      style={{ backgroundColor: "#004d1a" }}
      variant="dark"
      expand="lg"
      sticky="top"
    >
      <Container>
        <Navbar.Brand href="#" onClick={() => handleNavigation("/")}>
          HOTFormerLoc
        </Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            {navigationItems.map((item) =>
              item.children ? (
                <NavDropdown
                  title={
                    <span
                      onClick={(e) => {
                        e.preventDefault();
                        handleNavigation(item.path);
                      }}
                      style={{ cursor: "pointer" }}
                    >
                      {item.label}
                    </span>
                  }
                  id={`nav-dropdown-${item.label}`}
                  key={item.label}
                  show={openDropdown === item.label}
                  onMouseEnter={() => setOpenDropdown(item.label)}
                  onMouseLeave={() => setOpenDropdown(null)}
                >
                  {item.children.map((child) => (
                    <NavDropdown.Item
                      key={child.label}
                      onClick={() => handleNavigation(child.path)}
                      style={{ backgroundColor: "white", color: "#004d1a" }}
                    >
                      {child.label}
                    </NavDropdown.Item>
                  ))}
                </NavDropdown>
              ) : (
                <Nav.Link
                  key={item.label}
                  onClick={() => handleNavigation(item.path)}
                >
                  {item.label}
                </Nav.Link>
              )
            )}
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default NavBar;
