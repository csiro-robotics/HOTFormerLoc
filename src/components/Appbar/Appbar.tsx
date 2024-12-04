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
      { label: "Model Images", path: "/dataset#model-images" },
    ],
  },
  {
    label: "Dataset Visualisation",
    path: "/dataset-visualisation",
    children: [
      { label: "Overview", path: "/dataset-visualisation#overview" },
      {
        label: "Compare Visualisations of Forests",
        path: "/dataset-visualisation#compare",
      },
    ],
  },
  {
    label: "Download",
    path: "/download",
    children: [
      { label: "Download Link", path: "/download#download-link" },
      { label: "Usage Examples", path: "/download#usage-examples" },
    ],
  },
  { label: "Acknowledgements", path: "/acknowledgements" },
  { label: "Citation", path: "/citation" },
];

const Appbar = ({ siteName }: { siteName: string }) => {
  const navigate = useNavigate();
  const isSmallScreen = useMediaQuery("(max-width:1400px)");
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
            sx={{
              pl: depth * 2 + 1,
              backgroundColor: openItems.has(item.label)
                ? "rgba(0, 123, 255, 0.1)"
                : "transparent",
              "&:hover": {
                backgroundColor: "rgba(0, 123, 255, 0.15)",
              },
              transition: "background-color 0.3s ease",
            }}
          >
            <ListItemText
              primary={item.label}
              primaryTypographyProps={{
                fontWeight: openItems.has(item.label) ? "bold" : "normal",
                fontSize: "0.95rem",
                color: "#333",
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
    <Box sx={{ display: "flex" }}>
      {isSmallScreen ? (
        <>
          {
            {
              /* Responsiveness for Smaller Devices or screens*/
            }
          }
          <AppBar position="fixed" sx={{ backgroundColor: "white" }}>
            <Toolbar>
              <IconButton
                edge="start"
                aria-label="menu"
                onClick={() => toggleDrawer(true)}
              >
                <MenuIcon />
              </IconButton>
              <Typography
                variant="h6"
                sx={{ flexGrow: 1, fontWeight: "bold", color: "green" }}
              >
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
            <Box
              sx={{
                width: 250,
                backgroundColor: "#f4f5f7",
                height: "100%",
                boxSizing: "border-box",
              }}
            >
              <Typography
                variant="h6"
                sx={{ px: 2, py: 2, fontWeight: "bold", color: "green" }}
              >
                {siteName}
              </Typography>
              <List>{renderMenu(navigationItems)}</List>
            </Box>
          </SwipeableDrawer>
        </>
      ) : (
        // Sidebar for larger screens
        <Drawer
          variant="permanent"
          anchor="left"
          sx={{
            "& .MuiDrawer-paper": {
              width: 280,
              boxSizing: "border-box",
              backgroundColor: "#f4f5f7",
              borderRight: "1px solid #ddd",
            },
          }}
        >
          <Typography
            variant="h6"
            sx={{
              px: 2,
              py: 2,
              fontWeight: "bold",
              color: "green",
            }}
          >
            {siteName}
          </Typography>
          <List>{renderMenu(navigationItems)}</List>
        </Drawer>
      )}
      <Box
        sx={{
          flexGrow: 1,
          ml: isSmallScreen ? 0 : 280,
          mt: isSmallScreen ? 8 : 0,
        }}
      ></Box>
    </Box>
  );
};

export default Appbar;
