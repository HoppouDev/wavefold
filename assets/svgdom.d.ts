// svgdom has no types of its own, so the essential types are covered here
declare module "svgdom" {
  export function createSVGWindow(): Window & { document: Document };
}
