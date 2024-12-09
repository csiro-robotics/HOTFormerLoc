import React from "react";
import { useNavigate } from "react-router-dom";
import AppBar from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import Box from "@mui/material/Box";
import List from "@mui/material/List";
import ListItem from "@mui/material/ListItem";
import ListItemButton from "@mui/material/ListItemButton";
import ListItemText from "@mui/material/ListItemText";
import Collapse from "@mui/material/Collapse";
import ExpandLess from "@mui/icons-material/ExpandLess";
import ExpandMore from "@mui/icons-material/ExpandMore";
import Drawer from "@mui/material/Drawer";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import SwipeableDrawer from "@mui/material/SwipeableDrawer";
import useMediaQuery from "@mui/material/useMediaQuery";
import styles from "./Appbar.module.css"; // Import module.css

const navigationItems = [
  {
    label: "Paper",
    path: "/hotformerloc/",
    children: [
      { label: "Abstract", path: "/#abstract" },
      {
        label: "Network Architecture",
        path: "/#network-architecture",
        children: [
          { label: "HOTFormerLoc", path: "/#hotformerloc" },
          { label: "Cyndrical Octree Attention", path: "/#coa" },
          { label: "Pyramid Attention Pooling", path: "/#pap" },
        ],
      },
      {
        label: "Experiments",
        path: "/#experiments",
        children: [
          {
            label: "Datasets and Evaluation Criteria",
            path: "/#evaluation-criteria",
          },
          { label: "Ablation Study", path: "/#ablation-study" },
        ],
      },
      { label: "Future Work", path: "/#future-work" },
    ],
  },
  {
    label: "Dataset",
    path: "/dataset",
    children: [
      { label: "Overview", path: "/dataset#overview" },
      { label: "Data Collection Methodology", path: "/dataset#methodology" },
      { label: "Benchmarking", path: "/dataset#benchmarking" },
      { label: "Visualisation", path: "/dataset#visualisation" },
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
  { label: "Acknowledgements", path: "/acknowledgements" },
  { label: "Citation", path: "/citation" },
];

const Appbar = ({ siteName }: { siteName: string }) => {
  const navigate = useNavigate();
  const isSmallScreen = useMediaQuery("(max-width:1650px)");
  const [drawerOpen, setDrawerOpen] = React.useState(false);
  const [openItems, setOpenItems] = React.useState<Set<string>>(new Set());

  const handleNavigation = (path: string) => {
    const [pathname, hash] = path.split("#");

    if (pathname) {
      navigate(pathname, { replace: true });
    }

    if (hash) {
      setTimeout(() => {
        const element = document.getElementById(hash);
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 0);
    }
  };

  const toggleDrawer = (open: boolean) => {
    setDrawerOpen(open);
  };

  const toggleOpen = (label: string) => {
    const newOpenItems = new Set(openItems);
    if (newOpenItems.has(label)) {
      newOpenItems.delete(label);
    } else {
      newOpenItems.add(label);
    }
    setOpenItems(newOpenItems);
  };

  const renderMenu = (items: typeof navigationItems, depth = 0) =>
    items.map((item) => (
      <div key={item.label}>
        <ListItem disablePadding>
          <ListItemButton
            onClick={() => {
              if (item.children) {
                toggleOpen(item.label);
              } else {
                handleNavigation(item.path);
              }
            }}
            className={`${styles.listItemButton} ${
              openItems.has(item.label) ? styles.activeListItem : ""
            }`}
            style={{ paddingLeft: `${depth * 16 + 8}px` }}
          >
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                className: openItems.has(item.label)
                  ? styles.activeText
                  : styles.inactiveText,
              }}
            />
            {item.children &&
              (openItems.has(item.label) ? <ExpandLess /> : <ExpandMore />)}
          </ListItemButton>
        </ListItem>
        {item.children && (
          <Collapse in={openItems.has(item.label)} timeout="auto" unmountOnExit>
            <List component="div" disablePadding>
              {renderMenu(item.children, depth + 1)}
            </List>
          </Collapse>
        )}
      </div>
    ));

  return (
    <Box className={styles.container}>
      {isSmallScreen ? (
        <>
          <AppBar
            position="fixed"
            className={styles.appBar}
          >
            <Toolbar>
              <IconButton
                edge="start"
                aria-label="menu"
                onClick={() => toggleDrawer(true)}
              >
                <MenuIcon />
              </IconButton>
              <Typography variant="h6" className={styles.siteName}>
                {siteName}
              </Typography>
            </Toolbar>
          </AppBar>
          <SwipeableDrawer
            anchor="left"
            open={drawerOpen}
            onClose={() => toggleDrawer(false)}
            onOpen={() => toggleDrawer(true)}
          >
            <Box className={styles.drawer}>
              <Typography variant="h6" className={styles.drawerTitle}>
                {siteName}
              </Typography>
              <List>{renderMenu(navigationItems)}</List>
            </Box>
          </SwipeableDrawer>
        </>
      ) : (
        <Drawer variant="permanent" anchor="left" className={styles.drawer}>
          <Typography variant="h6" className={styles.drawerTitle}>
            {siteName}
          </Typography>
          <List>{renderMenu(navigationItems)}</List>
        </Drawer>
      )}
    </Box>
  );
};

export default Appbar;
