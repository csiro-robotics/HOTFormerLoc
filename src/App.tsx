import "./App.css";
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import DefaultPage from "./pages/DefaultPage";
import PaperPage from "./pages/Paper";
import DatasetPage from "./pages/Dataset";
import DownloadPage from "./pages/Download";
import NavigationBar from "./components/NavigationBar/NavigationBar";

const navigationItems: NavigationItem[] = [
  {
    label: "Home",
    path: "/",
    children: [
      { label: "Overview", path: "/#overview" },
      { label: "Interactive Submap Visualisation", path: "/#visualise-submap" },
      { label: "Citation", path: "/#citation" },
      { label: "Contact Us", path: "/#contact" },
      { label: "Updates", path: "/#updates" },
      { label: "Acknowledgements", path: "/#acknowledgements" },
    ],
  },
  {
    label: "HOTFormerLoc",
    path: "/paper",
    children: [
      { label: "Summary", path: "/paper#summary" },
      {
        label: "Network Architecture",
        path: "/paper#network-architecture",
      },
      {
        label: "Experiments",
        path: "/paper#experiments",
      },
      { label: "Future Work", path: "/paper#future-work" },
    ],
  },
  {
    label: "CS-Wild-Places",
    path: "/dataset",
    children: [
      { label: "Overview", path: "/dataset#overview" },
      { label: "Visualisation", path: "/dataset#visualisation" },
      { label: "Methodology", path: "/dataset#methodology" },
      { label: "Benchmarking", path: "/dataset#benchmarking" },
    ],
  },
  {
    label: "Downloads",
    path: "/download",
    children: [
      { label: "Checkpoints", path: "/download#checkpoint" },
      { label: "Dataset", path: "/download#dataset" },
      { label: "Usage Examples", path: "/download#usage-examples" },
    ],
  },
];

function App() {
  return (
    <Router basename="/HOTFormerLoc">
       <NavigationBar navigationItems={navigationItems} />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<DefaultPage/>} />
          <Route path="/paper" element={<PaperPage/>} />
          <Route path="/dataset" element={<DatasetPage/>} />
          <Route path="/download" element={<DownloadPage/>} />
        </Routes>
      </main>
    </Router>
  );
}

export default App;
