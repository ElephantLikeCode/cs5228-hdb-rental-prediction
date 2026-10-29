import numpy as np
import pandas as pd
from src.followup_analysis import grouped_risk, interchange_features


def test_risk_unknown_group_falls_back_to_prior_and_ignores_query_labels():
    fit=pd.DataFrame({'TOWN':['A','A'],'FLAT_TYPE':['3-room','3 room']})
    query=pd.DataFrame({'TOWN':['A','B'],'FLAT_TYPE':['3 room','4 room'],'MONTHLY_RENT':[1,99999]})
    result=grouped_risk(fit,np.array([True,False]),query)
    assert np.allclose(result,.5)
    query.MONTHLY_RENT=-1
    assert np.array_equal(result,grouped_risk(fit,np.array([True,False]),query))


def test_interchange_excludes_unannounced_service_and_counts_lines_not_platforms():
    calendar=pd.DataFrame({'CODE':['ew1','ns1','te1','ew2'],'NAME':['hub','hub','hub','hub'],
        'LATITUDE':[1.3]*4,'LONGITUDE':[103.8]*4,
        'effective':pd.to_datetime(['2021-01-01']*4),
        'announced':pd.to_datetime(['2021-01-01','2021-01-01','2025-02-01','2021-01-01']),
        'source':['operational_before_study','operational_before_study','event','operational_before_study']})
    query=pd.DataFrame({'RENT_APPROVAL_DATE':['2025-03']})
    core=pd.DataFrame({'LATITUDE':[1.3],'LONGITUDE':[103.8]})
    result=interchange_features(query,core,calendar,origin='2025-01')
    assert result.ACCESS_INTERCHANGE_KM.iloc[0]==0
    assert result.ACCESS_LINES_1KM.iloc[0]==2
    assert interchange_features(query,core,calendar).ACCESS_LINES_1KM.iloc[0]==3
