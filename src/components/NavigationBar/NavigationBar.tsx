import { FC, useState, MouseEvent, memo, useCallback } from "react";
import { Navbar, Container, Nav, NavDropdown } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import { HashLink } from "react-router-hash-link";
import "./NavigationBar.css"

const HoverNavDropdown: FC<{ item: NavigationItem }> = ({ item }) => {
  const [show, setShow] = useState(false);
  const navigate = useNavigate();

  const handleMouseEnter = useCallback(() => setShow(true), []);
  const handleMouseLeave = useCallback(() => setShow(false), []);

  const handleParentClick = (event: MouseEvent<HTMLSpanElement>) => {
    event.preventDefault();
    event.stopPropagation();
    navigate(item.path);
  };

  return (
    <NavDropdown
      title={
        <span onClick={handleParentClick} style={{ cursor: "pointer" }}>
          {item.label}
        </span>
      }
      id={`${item.label}-dropdown`}
      show={show}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {item.children?.map((child) => (
        <NavDropdown.Item
          as={HashLink}
          smooth
          to={child.path}
          key={child.label}
        >
          {child.label}
        </NavDropdown.Item>
      ))}
    </NavDropdown>
  );
};

const NavigationBar: FC<NavigationBarProps> = memo(({ navigationItems }) => {
  return (
    <Navbar
      style={{
        backgroundColor: "#004d1a"
      }}
      variant="dark"
      expand="lg"
      sticky="top"
    >
      <Container>
        <Navbar.Brand as={HashLink} to="/" smooth>
          HOTFormerLoc
        </Navbar.Brand>

        <Navbar.Toggle aria-controls="navbar-nav" />
        <Navbar.Collapse id="navbar-nav">
          <Nav className="ms-auto">
            {navigationItems.map((item) =>
              item.children && item.children.length > 0 ? (
                <HoverNavDropdown item={item} key={item.label} />
              ) : (
                <Nav.Link as={HashLink} smooth to={item.path} key={item.label}>
                  {item.label}
                </Nav.Link>
              )
            )}
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}, (prevProps, nextProps) => prevProps.navigationItems === nextProps.navigationItems);

export default NavigationBar;