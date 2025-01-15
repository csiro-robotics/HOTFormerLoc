interface Author {
  firstName: string;
  lastName: string;
  university: string;
  link: string;
}

interface AuthorsBarProps {
  authors: Author[];
}