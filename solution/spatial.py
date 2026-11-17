"""Finite team-set geometry: no player IDs or human possession fields."""
import numpy as np
from .evidence import validate_blocks


def spatial_features(tracking, blocks, fps, height=None):
    ball = np.asarray(tracking.ball, dtype=float); players = np.asarray(tracking.players, dtype=float)
    team = np.asarray(tracking.team); n = len(ball)
    validate_blocks(blocks, n)
    if ball.shape != (n, 2) or players.shape != (n, len(team), 2) or not np.isin(team, [0, 1]).all():
        raise ValueError('invalid spatial observation shape')
    if not np.isfinite(fps) or fps <= 0 or np.isinf(ball).any() or np.isinf(players).any():
        raise ValueError('invalid spatial coordinates or cadence')
    pitch = np.asarray(tracking.pitch, dtype=float)
    if pitch.shape != (2,) or not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ValueError('invalid pitch')
    columns = {}; valid = np.isfinite(ball).all(axis=1)
    columns['ball_valid'] = valid.astype(float)
    centred = np.abs(ball - pitch / 2) / (pitch / 2)
    columns['ball_longitudinal_abs'] = np.where(valid, np.minimum(centred[:, 0], 2), 0)
    columns['ball_lateral_abs'] = np.where(valid, np.minimum(centred[:, 1], 2), 0)
    columns['ball_outside'] = valid & ((ball < 0).any(axis=1) | (ball > pitch).any(axis=1))
    velocity = np.zeros((n, 2)); velocity_valid = np.zeros(n, bool)
    inferred_speed = np.full(players.shape[:2], np.nan)
    for start, stop in blocks:
        if stop - start < 2:
            continue
        delta = np.diff(ball[start:stop], axis=0) * fps
        okay = np.isfinite(delta).all(axis=1) & (np.linalg.norm(delta, axis=1) <= 45)
        velocity[start + 1:stop] = np.where(okay[:, None], delta, 0)
        velocity_valid[start + 1:stop] = okay
        player_speed = np.linalg.norm(np.diff(players[start:stop], axis=0), axis=2) * fps
        inferred_speed[start + 1:stop] = np.where(player_speed <= 15, player_speed, np.nan)
    columns['ball_speed'] = np.linalg.norm(velocity, axis=1) / 45
    columns['ball_velocity_valid'] = velocity_valid.astype(float)
    acceleration = np.zeros(n)
    for start, stop in blocks:
        if stop - start > 1:
            okay = velocity_valid[start + 1:stop] & velocity_valid[start:stop - 1]
            acceleration[start + 1:stop] = np.where(okay, np.minimum(np.linalg.norm(np.diff(velocity[start:stop], axis=0), axis=1) * fps, 50) / 50, 0)
    columns['ball_acceleration'] = acceleration
    if height is None:
        height = np.full(n, np.nan)
    height = np.asarray(height)
    if height.shape != (n,):
        raise ValueError('invalid ball height shape')
    columns['ball_height_valid'] = valid & np.isfinite(height)
    columns['ball_height'] = np.where(valid & np.isfinite(height), np.clip(height, 0, 10) / 10, 0)
    supplied_speed = np.asarray(tracking.speed, dtype=float)
    if supplied_speed.shape != players.shape[:2]:
        raise ValueError('invalid player speed shape')
    speed = np.where(np.isfinite(supplied_speed) & (supplied_speed >= 0) & (supplied_speed <= 15), supplied_speed, inferred_speed)
    for code, side in enumerate(('home', 'away')):
        points = players[:, team == code]; visible = np.isfinite(points).all(axis=2)
        count = visible.sum(axis=1)
        columns[side + '_visible_fraction'] = np.minimum(count, 22) / 22
        distance = np.linalg.norm(points - ball[:, None, :], axis=2)
        distance[~np.isfinite(distance)] = np.inf
        ordered = np.sort(distance, axis=1)
        for k in range(1, 4):
            values = ordered[:, k - 1] if ordered.shape[1] >= k else np.full(n, np.inf)
            columns[f'{side}_nearest_{k}'] = np.minimum(values, 40) / 40
        for radius in (2, 5, 10):
            columns[f'{side}_within_{radius}m'] = (distance < radius).sum(axis=1) / 22
        centroid = np.nansum(np.where(visible[:, :, None], points, np.nan), axis=1) / np.maximum(count, 1)[:, None]
        columns[side + '_centroid_distance'] = np.where(valid & (count > 0), np.minimum(np.linalg.norm(centroid - ball, axis=1), 100) / 100, 0)
        deviation = np.where(visible[:, :, None], points - centroid[:, None, :], 0)
        spread = np.sqrt(np.sum(deviation**2, axis=1) / np.maximum(count, 1)[:, None]) / pitch
        columns[side + '_spread_length'] = spread[:, 0]
        columns[side + '_spread_width'] = spread[:, 1]
        values = speed[:, team == code]; finite = np.isfinite(values) & visible
        columns[side + '_speed_mean'] = np.where(finite, values, 0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1) / 15
        columns[side + '_speed_coverage'] = finite.sum(axis=1) / 22
    result = np.column_stack(list(columns.values())).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('non-finite derived spatial features')
    return result, tuple(columns)
