import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchProjects, fetchExperiments, fetchRuns } from '../lib/api';

// Mock fetch globally
global.fetch = vi.fn();

describe('API Library', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('fetchProjects retrieves and parses data', async () => {
    const mockData = [{ id: 1, name: 'Test Project', created_at: '2024-01-01T00:00:00Z' }];
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });

    const data = await fetchProjects();
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/projects/'), expect.any(Object));
    expect(data).toEqual(mockData);
  });

  it('fetchExperiments retrieves and parses data', async () => {
    const mockData = [{ id: 1, project_id: 1, name: 'Exp 1' }];
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });

    const data = await fetchExperiments(1);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/projects/1/experiments'), expect.any(Object));
    expect(data).toEqual(mockData);
  });

  it('fetchRuns retrieves and parses metrics correctly', async () => {
    const mockData = [
      { id: 10, metrics: { test_wer: 0.15, train_loss: 1.2 } }
    ];
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockData,
    });

    const data = await fetchRuns(5);
    expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/experiments/5/runs'), expect.any(Object));
    expect(data[0].metrics.test_wer).toBe(0.15);
  });

  it('throws an error on failed response', async () => {
    (global.fetch as any).mockResolvedValue({
      ok: false,
      status: 500,
    });

    await expect(fetchProjects()).rejects.toThrow('Failed to fetch projects');
  });
});
