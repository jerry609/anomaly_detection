import pandas as pd
import numpy as np
from collections import defaultdict
from ..models.base_detector import *


class AccessAnalyzer:
    """
    Analyzes user access patterns for suspicious behavior.
    Detects unusual resource access, privilege escalation, etc.
    """

    def __init__(self, detector_model=None):
        """
        Initialize the access analyzer with a detector model.

        Args:
            detector_model (BaseDetector, optional): Model used for anomaly detection.
                                                    If None, a default model will be used.
        """
        self.detector = detector_model
        self.user_access_profiles = {}
        self.resource_sensitivity = {}  # Store sensitivity levels of resources

    def set_resource_sensitivity(self, resource_dict):
        """
        Set sensitivity levels for resources.

        Args:
            resource_dict (dict): Mapping of resource_id to sensitivity level (0-1)
        """
        self.resource_sensitivity = resource_dict

    def fit(self, access_data, device_data=None, ldap_data=None):
        """
        Build access profiles for users based on historical data.

        Args:
            access_data (pd.DataFrame): DataFrame containing access events
            device_data (pd.DataFrame, optional): Device-related information
            ldap_data (pd.DataFrame, optional): LDAP/permission information

        Returns:
            self: Returns the instance itself
        """
        # Group by user and build access profiles
        for user_id, user_data in access_data.groupby('user_id'):
            # Track commonly accessed resources
            resource_counts = user_data['resource_id'].value_counts()
            frequent_resources = resource_counts[resource_counts > 2].index.tolist()

            # Build profile
            self.user_access_profiles[user_id] = {
                'frequent_resources': frequent_resources,
                'access_count_avg': user_data.groupby('date')['resource_id'].count().mean(),
                'access_count_std': user_data.groupby('date')['resource_id'].count().std(),
                'num_unique_resources': len(user_data['resource_id'].unique()),
                'department': user_data['department'].iloc[0] if 'department' in user_data.columns else None
            }

            # Add role information if LDAP data is provided
            if ldap_data is not None and 'user_id' in ldap_data.columns:
                user_ldap = ldap_data[ldap_data['user_id'] == user_id]
                if not user_ldap.empty:
                    self.user_access_profiles[user_id]['roles'] = user_ldap['role'].tolist()
                    self.user_access_profiles[user_id]['groups'] = user_ldap['group'].tolist()

        # Train the detector if provided
        if self.detector:
            # Extract features for training
            features = self._extract_access_profile_features(access_data)
            self.detector.fit(features)

        return self

    def _extract_access_profile_features(self, data):
        """Extract access-related features for anomaly detection."""
        features = pd.DataFrame()

        # Calculate number of accesses per day for each user
        daily_access = data.groupby(['user_id', 'date']).size().reset_index(name='daily_access_count')

        # Calculate number of unique resources accessed per day for each user
        daily_unique = data.groupby(['user_id', 'date'])['resource_id'].nunique().reset_index(
            name='daily_unique_resources')

        # Merge features
        features = pd.merge(daily_access, daily_unique, on=['user_id', 'date'])

        # Add resource sensitivity if available
        if self.resource_sensitivity and 'resource_id' in data.columns:
            data['resource_sensitivity'] = data['resource_id'].map(
                lambda x: self.resource_sensitivity.get(x, 0.5))
            max_sensitivity = data.groupby(['user_id', 'date'])['resource_sensitivity'].max().reset_index()
            features = pd.merge(features, max_sensitivity, on=['user_id', 'date'])

        return features

    def detect_anomalies(self, new_access_data):
        """
        Detect access-based anomalies in new events.

        Args:
            new_access_data (pd.DataFrame): New access events to analyze

        Returns:
            pd.DataFrame: Anomaly scores and flags for each event
        """
        results = []

        # Group events by user and date for contextual analysis
        for (user_id, date), events in new_access_data.groupby(['user_id', 'date']):
            # Skip if we don't have a profile for this user
            if user_id not in self.user_access_profiles:
                for idx, event in events.iterrows():
                    results.append({
                        'event_id': idx,
                        'user_id': user_id,
                        'resource_id': event.get('resource_id', 'unknown'),
                        'date': date,
                        'anomaly_score': 0,
                        'is_anomaly': False,
                        'reason': "No user profile exists"
                    })
                continue

            profile = self.user_access_profiles[user_id]
            daily_access_count = len(events)
            unique_resources = events['resource_id'].nunique()

            # Check anomalies for each event
            for idx, event in events.iterrows():
                anomaly_score = 0
                anomaly_reason = []
                resource_id = event.get('resource_id', 'unknown')

                # Check for access to unusual resources
                if resource_id not in profile['frequent_resources']:
                    anomaly_score += 0.4
                    anomaly_reason.append(f"Unusual resource access: {resource_id}")

                # Check for abnormal number of daily accesses
                if profile['access_count_std'] > 0:
                    z_score = (daily_access_count - profile['access_count_avg']) / profile['access_count_std']
                    if z_score > 2:
                        anomaly_score += 0.3
                        anomaly_reason.append(f"Abnormally high access frequency: {daily_access_count} accesses")

                # Check for abnormal number of unique resources
                if unique_resources > 2 * profile['num_unique_resources']:
                    anomaly_score += 0.5
                    anomaly_reason.append(
                        f"Accessing unusually diverse resources: {unique_resources} different resources")

                # Check resource sensitivity
                if resource_id in self.resource_sensitivity:
                    sensitivity = self.resource_sensitivity[resource_id]
                    if sensitivity > 0.7:
                        anomaly_score += sensitivity * 0.3
                        anomaly_reason.append(f"Accessing high-sensitivity resource: {sensitivity:.2f}")

                # Use detector model if available
                if self.detector and len(events) > 0:
                    features = self._extract_access_profile_features(pd.DataFrame([event]))
                    if not features.empty:
                        model_score = self.detector.score(features)[0]
                        anomaly_score = max(anomaly_score, model_score)

                results.append({
                    'event_id': idx,
                    'user_id': user_id,
                    'resource_id': resource_id,
                    'date': date,
                    'anomaly_score': anomaly_score,
                    'is_anomaly': anomaly_score > 0.7,  # Threshold
                    'reason': ", ".join(anomaly_reason) if anomaly_reason else "Normal behavior"
                })

        return pd.DataFrame(results)
