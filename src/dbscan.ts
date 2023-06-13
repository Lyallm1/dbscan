import { euclidean } from 'ml-distance-euclidean';

interface IPoint {
  value: number[];
  index: number;
  label: number;
}
export interface DBScanOptions {
  epsilon?: number;
  minPoints?: number;
  distance?: (p: number[], q: number[]) => number;
}

export function dbscan(points: number[][], options: DBScanOptions = {}) {
  if (!(points instanceof Array)) throw Error(`points must be of type array, ${typeof points} given`);
  const { epsilon = 1, minPoints = 2, distance = euclidean } = options, data: IPoint[] = [], labels: number[] = [];
  let clusterId = 0;
  points.forEach((point, i) => data.push({ index: i, value: point, label: -1 }));
  for (const point of data) {
    if (point.label !== -1) {
      return;
    }
    let neighbors = rangeQuery(point, data, epsilon, distance);
    if (neighbors.length < minPoints) {
      point.label = 0;
      return;
    }
    clusterId++;
    point.label = clusterId;
    let neighbors2 = neighbors.filter(neighbor => neighbor.index !== point.index);
    while (neighbors2.length) {
      const neighbor = neighbors2.pop();
      if (!neighbor) break;
      if (neighbor.label === 0) neighbor.label = clusterId;
      if (neighbor.label !== -1) continue;
      neighbor.label = clusterId;
      neighbors = rangeQuery(neighbor, data, epsilon, distance);
      if (neighbors.length >= minPoints) neighbors2 = neighbors2.concat(neighbors);
    }
    labels.push(point.label - 1);
  }
  return labels;
}
function rangeQuery(current: IPoint, data: IPoint[], epsilon: number, distance: (p: number[], q: number[]) => number) {
  return data.filter(point => distance(point.value, current.value) <= epsilon);
}
